from __future__ import annotations

import asyncio
import logging
from dotenv import load_dotenv
import json
import os
import time
import wave
import re
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

from livekit import rtc, api
from livekit.agents import (
    AgentSession,
    Agent,
    JobContext,
    function_tool,
    RunContext,
    get_job_context,
    cli,
    WorkerOptions,
    RoomInputOptions,
)
from livekit.plugins import (
    deepgram,
    google,
    cartesia,
    silero,
    noise_cancellation,
)
from livekit.plugins.turn_detector.english import EnglishModel

# Load environment variables
load_dotenv(dotenv_path=".env.local")
logger = logging.getLogger("debt-collector")
logger.setLevel(logging.INFO)

outbound_trunk_id = os.getenv("SIP_OUTBOUND_TRUNK_ID")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Database simulation
@dataclass
class DebtRecord:
    phone_number: str
    debtor_name: str = "John Doe"
    debt_amount: float = 1500.00
    due_date: str = "2023-11-30"
    account_number: str = ""
    call_history: List[Dict[str, Any]] = None
    payment_plan: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if not self.account_number:
            self.account_number = f"ACCT-{self.phone_number[-4:]}"
        if self.call_history is None:
            self.call_history = []

class DebtDatabase:
    _records: Dict[str, DebtRecord] = {}
    
    @classmethod
    def get_record(cls, phone_number: str) -> DebtRecord:
        if phone_number not in cls._records:
            cls._records[phone_number] = DebtRecord(phone_number=phone_number)
        return cls._records[phone_number]
    
    @classmethod
    def add_call_record(
        cls, 
        phone_number: str, 
        recording_path: str,
        outcome: str,
        conversation: List[Tuple[str, str]]
    ):
        record = cls.get_record(phone_number)
        record.call_history.append({
            "timestamp": int(time.time()),
            "recording": recording_path,
            "outcome": outcome,
            "conversation": conversation
        })
    
    @classmethod
    def record_payment_plan(
        cls,
        phone_number: str,
        amount: float,
        installments: int,
        start_date: str
    ):
        record = cls.get_record(phone_number)
        record.payment_plan = {
            "amount": amount,
            "installments": installments,
            "start_date": start_date
        }


class CallRecorder:
    def __init__(self, output_path: str):
        self._output_path = output_path
        self._file = wave.open(output_path, 'wb')
        self._file.setnchannels(1)
        self._file.setsampwidth(2)
        self._file.setframerate(16000)
        self._sink = None  # Will be set when we get the audio track
        
    def attach_to_track(self, track):
        """Attach recorder to an audio track"""
        self._sink = rtc.AudioSink(track)
        self._sink.on("frame_received", self.on_audio_frame)
        
    def on_audio_frame(self, frame: rtc.AudioFrame):
        """Write raw audio data to WAV file"""
        try:
            self._file.writeframesraw(frame.data)
        except Exception as e:
            logger.error(f"Error writing audio frame: {e}")
    
    async def stop(self):
        """Stop recording and close file"""
        if self._sink:
            # Disconnect the sink
            self._sink.off("frame_received", self.on_audio_frame)
        self._file.close()
        logger.info(f"Recording saved to: {self._output_path}")


class DebtCollectionAgent(Agent):
    def __init__(
        self,
        *,
        debt_record: DebtRecord,
        dial_info: dict[str, Any],
        test_mode: bool = False,
        instructions: Optional[str] = None
    ):
        self.test_mode = test_mode
        self.original_instructions = f"""
        You are a professional debt collection agent for Riverline Financial Services. 
        You are calling {debt_record.debtor_name} (Account: {debt_record.account_number}) 
        about an overdue payment of ${debt_record.debt_amount:.2f} due since {debt_record.due_date}.
        
        GUIDELINES:
        1. Be firm but polite and professional
        2. State purpose of call upfront
        3. Verify identity by asking for last 4 digits of account number
        4. Offer payment options: full payment, payment plan, or dispute
        5. Document commitments using record_payment tool
        6. Transfer to human agent when requested using transfer_call
        7. End call professionally after resolution
        8. For voicemails, leave brief message with callback info
        """
        
        # Use custom instructions if provided, otherwise use original
        final_instructions = instructions if instructions else self.original_instructions
        
        super().__init__(instructions=final_instructions)
        self.debt_record = debt_record
        self.dial_info = dial_info
        self.recorder: Optional[CallRecorder] = None
        self.call_outcome = "unresolved"
        self.conversation_history: List[Tuple[str, str]] = []  # (speaker, text)
        
        # For runtime self-correction
        self.llm_system_prompt = final_instructions  # Dynamic system prompt
        self.self_correction_count = 0
        self.max_self_corrections = 3

    def set_recorder(self, recorder: CallRecorder):
        self.recorder = recorder

    async def finalize_call(self, outcome: str):
        """Finalize call and save recording"""
        self.call_outcome = outcome
        if self.recorder:
            await self.recorder.stop()
            DebtDatabase.add_call_record(
                self.debt_record.phone_number,
                self.recorder._output_path,
                outcome,
                self.conversation_history
            )
            logger.info(f"Saved recording: {self.recorder._output_path}")

    async def hangup(self):
        """End the call by deleting the room"""
        job_ctx = get_job_context()
        await job_ctx.api.room.delete_room(
            api.DeleteRoomRequest(room=job_ctx.room.name)
        )
        await self.finalize_call(self.call_outcome)

    @function_tool()
    async def transfer_call(self, ctx: RunContext):
        """Transfer call to human agent after confirmation"""
        transfer_to = self.dial_info.get("transfer_to")
        if not transfer_to:
            return "cannot transfer call"

        await ctx.session.generate_reply(
            instructions="politely inform you're transferring them"
        )

        job_ctx = get_job_context()
        try:
            await job_ctx.api.sip.transfer_sip_participant(
                api.TransferSIPParticipantRequest(
                    room_name=job_ctx.room.name,
                    participant_identity=ctx.session.participant.identity,
                    transfer_to=f"tel:{transfer_to}",
                )
            )
            await self.finalize_call("transferred")
        except Exception as e:
            logger.error(f"Transfer error: {e}")
            await ctx.session.generate_reply(
                instructions="apologize for the transfer issue"
            )
            await self.hangup()

    @function_tool()
    async def end_call(self, ctx: RunContext):
        """End the call professionally"""
        await ctx.session.generate_reply(
            instructions="thank them and end the call professionally"
        )
        await self.hangup()

    @function_tool()
    async def record_payment(
        self,
        ctx: RunContext,
        amount: float,
        payment_date: str,
        payment_method: str,
    ):
        """Record a payment commitment"""
        logger.info(f"Recording payment: ${amount} on {payment_date}")
        DebtDatabase.record_payment_plan(
            self.debt_record.phone_number,
            amount,
            1,  # Single installment for full payment
            payment_date
        )
        await self.finalize_call("payment_arranged")
        return f"${amount} payment recorded for {payment_date}"

    @function_tool()
    async def record_payment_plan(
        self,
        ctx: RunContext,
        amount: float,
        installments: int,
        start_date: str,
    ):
        """Record a payment plan commitment"""
        logger.info(f"Recording payment plan: ${amount} in {installments} installments starting {start_date}")
        DebtDatabase.record_payment_plan(
            self.debt_record.phone_number,
            amount,
            installments,
            start_date
        )
        await self.finalize_call("payment_plan_arranged")
        return f"Payment plan recorded: ${amount} in {installments} installments"

    @function_tool()
    async def detected_answering_machine(self, ctx: RunContext):
        """Handle voicemail detection"""
        await ctx.session.generate_reply(
            instructions=f"""
            Leave brief voicemail:
            1. Identify as Riverline Financial Services
            2. Mention account ending in {self.debt_record.account_number[-4:]}
            3. Request callback to {self.dial_info.get('callback_number', '1-800-RIVERLINE')}
            """
        )
        await self.finalize_call("voicemail_left")
        await self.hangup()
    
    
    async def process_response(self, text: str):
        """Track conversation and perform self-correction checks"""
        self.conversation_history.append(("agent", text))
        
        # Skip self-correction in test mode
        if self.test_mode:
            return
        
        # Only self-correct up to max times
        if self.self_correction_count >= self.max_self_corrections:
            return
        
        # Detect issues and apply corrections
        issues = self._detect_conversation_issues()
        if issues:
            await self._apply_runtime_correction(issues)

    def _detect_conversation_issues(self) -> Dict[str, Any]:
        """Analyze conversation and detect issues"""
        agent_text = " ".join([text for speaker, text in self.conversation_history if speaker == "agent"])
        issues = {}
        
        # 1. Repetition detection
        issues["repetition"] = self._calculate_repetition_score(agent_text)
        
        # 2. Negotiation detection
        issues["negotiation_attempts"] = self._count_negotiation_attempts(agent_text)
        
        # 3. Empathy score
        issues["empathy"] = self._calculate_empathy_score(agent_text)
        
        # 4. Resolution progress
        issues["resolution"] = self._assess_resolution_progress()
        
        return issues

    async def _apply_runtime_correction(self, issues: Dict[str, Any]):
        """Apply corrections based on detected issues"""
        # Build correction prompt
        correction = "Based on conversation analysis, adjust your approach:\n"
        
        # Add specific corrections based on issues
        if issues["repetition"] > 0.7:
            correction += "- Avoid repeating the same phrases\n"
        
        if issues["negotiation_attempts"] < 2:
            correction += "- Offer more payment options\n"
        
        if issues["empathy"] < 0.5:
            correction += "- Show more empathy and understanding\n"
            
        if issues["resolution"] == "stalled":
            correction += "- More assertively guide toward resolution\n"
        
        # Update system prompt
        self.llm_system_prompt = self.original_instructions + "\n\n" + correction
        self.self_correction_count += 1
        logger.info(f"Applied runtime correction: {correction}")

    def _calculate_repetition_score(self, text: str) -> float:
        """Calculate repetition score (0-1)"""
        sentences = [s.strip() for s in re.split(r'[.!?]', text) if s.strip()]
        if len(sentences) < 2:
            return 0.0
            
        vectorizer = TfidfVectorizer().fit_transform(sentences)
        similarities = cosine_similarity(vectorizer)
        return np.mean(similarities[np.triu_indices_from(similarities, k=1)])
    
    def _count_negotiation_attempts(self, text: str) -> int:
        """Count negotiation-related keywords"""
        keywords = ["payment plan", "installment", "settlement", 
                    "arrangement", "option", "reduce", "lower", "offer"]
        return sum(text.lower().count(kw) for kw in keywords)
    
    def _calculate_empathy_score(self, text: str) -> float:
        """Calculate empathy score (0-1)"""
        empathic_phrases = [
            "understand", "sorry to hear", "difficult time", 
            "appreciate", "challenging", "help you", 
            "work with you", "solution together"
        ]
        count = sum(text.lower().count(phrase) for phrase in empathic_phrases)
        return min(1.0, count / 3)  # Normalize to 0-1 scale
    
    def _assess_resolution_progress(self) -> str:
        """Assess progress toward resolution"""
        # Check if call is already resolved
        if self.call_outcome != "unresolved":
            return "resolved"
            
        # Check if payment arrangements were discussed
        if any("payment" in text for _, text in self.conversation_history):
            return "progressing"
            
        return "stalled"

# AI-Automated Testing Platform
class VoiceAgentTester:
    def __init__(self):
        self.llm = google.LLM(model="gemini-1.5-flash")
        self.vectorizer = TfidfVectorizer()
        self.personas = self._load_personas()
    
    def _load_personas(self) -> List[Dict[str, Any]]:
        """Generate customer personas for testing"""
        return [
            {"name": "Avoidant Andy", "description": "Tries to avoid the conversation, makes excuses"},
            {"name": "Aggressive Alex", "description": "Gets angry and confrontational about the debt"},
            {"name": "Hardship Helen", "description": "Genuine financial difficulties, needs payment plan"},
            {"name": "Negotiator Nate", "description": "Tries to negotiate lower settlement amount"},
            {"name": "Silent Sam", "description": "Minimal responses, hard to engage"},
            {"name": "Forgetful Fred", "description": "Claims to have forgotten about the debt"}
        ]
    
    async def generate_conversation(
        self, 
        agent_instructions: str,
        persona: Dict[str, Any]
    ) -> Tuple[List[Tuple[str, str]], Dict[str, float]]:
        """Simulate conversation between agent and persona"""
        conversation = []
        
        # System prompts
        agent_system = f"{agent_instructions}\n\nYou are talking to: {persona['name']} - {persona['description']}"
        persona_system = f"You are {persona['name']} - {persona['description']}. You're speaking with a debt collection agent."
        
        # Start conversation
        response = await self.llm.chat(
            messages=[{"role": "system", "content": agent_system},
                      {"role": "user", "content": "Start the conversation"}],
            max_tokens=200
        )
        agent_msg = response.choices[0].message.content
        conversation.append(("agent", agent_msg))
        
        # Simulate 5 turns
        for _ in range(5):
            # Persona responds
            response = await self.llm.chat(
                messages=[{"role": "system", "content": persona_system},
                          {"role": "user", "content": agent_msg}],
                max_tokens=200
            )
            persona_msg = response.choices[0].message.content
            conversation.append(("persona", persona_msg))
            
            # Agent responds
            response = await self.llm.chat(
                messages=[{"role": "system", "content": agent_system}] + 
                         [{"role": "user" if s == "persona" else "assistant", "content": m} 
                          for s, m in conversation],
                max_tokens=200
            )
            agent_msg = response.choices[0].message.content
            conversation.append(("agent", agent_msg))
        
        # Evaluate conversation
        metrics = self._evaluate_conversation(conversation)
        
        return conversation, metrics
    
    def _evaluate_conversation(
        self, 
        conversation: List[Tuple[str, str]]
    ) -> Dict[str, float]:
        """Evaluate conversation against key metrics"""
        agent_text = " ".join([text for speaker, text in conversation if speaker == "agent"])
        persona_text = " ".join([text for speaker, text in conversation if speaker == "persona"])
        
        # 1. Repetition Score
        sentences = re.split(r'[.!?]', agent_text)
        repetition_score = 0
        if len(sentences) > 1:
            vectors = self.vectorizer.fit_transform(sentences)
            similarities = cosine_similarity(vectors)
            repetition_score = np.mean(similarities[np.triu_indices_from(similarities, k=1)])
        
        # 2. Negotiation Effectiveness
        negotiation_keywords = ["plan", "settle", "option", "reduce", "lower", "arrangement"]
        negotiation_attempts = sum(1 for kw in negotiation_keywords if kw in agent_text.lower())
        
        # 3. Response Relevance
        agent_vec = self.vectorizer.transform([agent_text])
        persona_vec = self.vectorizer.transform([persona_text])
        relevance_score = cosine_similarity(agent_vec, persona_vec)[0][0] if agent_text and persona_text else 0.0
        
        return {
            "repetition": float(repetition_score),
            "negotiation_attempts": negotiation_attempts,
            "relevance": float(relevance_score),
            "success": 1.0 if negotiation_attempts > 2 and relevance_score > 0.5 else 0.0
        }
    
    async def test_agent(
        self, 
        agent_instructions: str,
        num_tests: int = 5
    ) -> Dict[str, Any]:
        """Run automated tests against multiple personas"""
        test_results = []
        all_metrics = []
        
        for _ in range(num_tests):
            persona = random.choice(self.personas)
            conversation, metrics = await self.generate_conversation(agent_instructions, persona)
            test_results.append({
                "persona": persona,
                "conversation": conversation,
                "metrics": metrics
            })
            all_metrics.append(metrics)
        
        # Aggregate metrics
        avg_repetition = np.mean([m["repetition"] for m in all_metrics]) if all_metrics else 0.0
        avg_negotiation = np.mean([m["negotiation_attempts"] for m in all_metrics]) if all_metrics else 0.0
        avg_relevance = np.mean([m["relevance"] for m in all_metrics]) if all_metrics else 0.0
        success_rate = np.mean([m["success"] for m in all_metrics]) if all_metrics else 0.0
        
        return {
            "tests": test_results,
            "summary": {
                "avg_repetition": avg_repetition,
                "avg_negotiation": avg_negotiation,
                "avg_relevance": avg_relevance,
                "success_rate": success_rate
            }
        }
    
    async def self_correct_agent(
        self, 
        agent_instructions: str,
        max_iterations: int = 5,
        success_threshold: float = 0.8
    ) -> str:
        """Automatically improve agent through testing and self-correction"""
        current_instructions = agent_instructions
        iteration = 0
        success_rate = 0.0
        
        while iteration < max_iterations and success_rate < success_threshold:
            # Run test suite
            test_results = await self.test_agent(current_instructions)
            success_rate = test_results["summary"]["success_rate"]
            
            logger.info(f"Iteration {iteration+1} - Success Rate: {success_rate:.2f}")
            
            if success_rate >= success_threshold:
                logger.info("Success threshold reached!")
                break
                
            # Generate improvements
            improvement_prompt = f"""
            Original Agent Instructions:
            {agent_instructions}
            
            Test Results:
            {json.dumps(test_results['summary'], indent=2)}
            
            Based on these results, rewrite the agent instructions to improve:
            1. Reduce repetition (current score: {test_results['summary']['avg_repetition']:.2f})
            2. Increase negotiation attempts (current: {test_results['summary']['avg_negotiation']:.2f})
            3. Improve response relevance (current: {test_results['summary']['avg_relevance']:.2f})
            
            Focus on these areas while maintaining professionalism.
            """
            
            response = await self.llm.chat(
                messages=[{"role": "user", "content": improvement_prompt}],
                max_tokens=1000
            )
            current_instructions = response.choices[0].message.content
            iteration += 1
            
            logger.info(f"New Instructions:\n{current_instructions}")
        
        return current_instructions

async def automated_testing_entry():
    """Entry point for automated testing and self-correction"""
    logger.info("Starting automated testing and self-correction process")
    
    # Initialize tester
    tester = VoiceAgentTester()
    
    # Base agent instructions
    base_instructions = """
    You are a professional debt collection agent. You are calling about an overdue payment.
    Guidelines:
    1. Be firm but polite
    2. State purpose upfront
    3. Verify identity
    4. Offer payment options
    5. Document commitments
    """
    
    # Run self-correction process
    improved_instructions = await tester.self_correct_agent(
        base_instructions,
        max_iterations=5,
        success_threshold=0.8
    )
    
    logger.info("Final improved instructions:")
    logger.info(improved_instructions)
    
    # Save improved instructions
    with open("improved_instructions.txt", "w") as f:
        f.write(improved_instructions)
    
    logger.info("Self-correction process completed")

async def agent_entrypoint(ctx: JobContext):
    """Entry point for the debt collection agent"""
    logger.info(f"Job received: ID={ctx.job.id}, Metadata={ctx.job.metadata}")
    await ctx.connect()
    
    if not ctx.job.metadata:
        logger.error("No metadata provided!")
        return
    
    try:
        dial_info = json.loads(ctx.job.metadata)
        phone_number = dial_info["phone_number"]
    except (json.JSONDecodeError, KeyError) as e:
        logger.error(f"Invalid metadata: {e}")
        return
    
    # Get debt record from database
    debt_record = DebtDatabase.get_record(phone_number)
    
    # Define base instructions
    base_instructions = f"""
    You are a professional debt collection agent for Riverline Financial Services. 
    You are calling {debt_record.debtor_name} (Account: {debt_record.account_number}) 
    about an overdue payment of ${debt_record.debt_amount:.2f} due since {debt_record.due_date}.
    
    GUIDELINES:
    1. Be firm but polite and professional
    2. State purpose of call upfront
    3. Verify identity by asking for last 4 digits of account number
    4. Offer payment options: full payment, payment plan, or dispute
    5. Document commitments using record_payment tool
    6. Transfer to human agent when requested using transfer_call
    7. End call professionally after resolution
    8. For voicemails, leave brief message with callback info
    """
    
    # Load improved instructions if available
    agent_instructions = base_instructions
    if os.path.exists("improved_instructions.txt"):
        with open("improved_instructions.txt", "r") as f:
            agent_instructions = f.read()
    
    # Create agent with debt record AND instructions
    agent = DebtCollectionAgent(
        debt_record=debt_record,
        dial_info=dial_info,
        test_mode=False,
        instructions=agent_instructions  # Pass instructions during creation
    )

    # Setup AI components
    session = AgentSession(
        turn_detection=EnglishModel(),
        vad=silero.VAD.load(),
        stt=deepgram.STT(),
        tts=cartesia.TTS(),
        llm=google.LLM(model="gemini-2.0-flash"),
    )

    # Start session before dialing
    session_task = asyncio.create_task(
        session.start(
            agent=agent,
            room=ctx.room,
            room_input_options=RoomInputOptions(
                noise_cancellation=noise_cancellation.BVCTelephony(),
            ),
        )
    )

    try:
        # Setup call recording before dialing
        timestamp = int(time.time())
        recording_path = f"recordings/{phone_number}_{timestamp}.wav"
        os.makedirs(os.path.dirname(recording_path), exist_ok=True)
        
        # Create recorder
        recorder = CallRecorder(recording_path)
        agent.set_recorder(recorder)
        
        # Set up participant connection handler for recording
        recording_started = False
        
        def on_participant_connected(participant):
            nonlocal recording_started
            if not recording_started and participant.identity == phone_number:
                logger.info(f"Participant {participant.identity} connected, setting up recording")
                
                # Look for audio track from the participant
                for track_pub in participant.track_publications.values():
                    if track_pub.track and track_pub.track.kind == rtc.TrackKind.KIND_AUDIO:
                        recorder.attach_to_track(track_pub.track)
                        recording_started = True
                        logger.info(f"Recording started: {recording_path}")
                        break
                
                # If no track available yet, wait for track subscription
                if not recording_started:
                    def on_track_subscribed(track, publication, participant):
                        nonlocal recording_started
                        if not recording_started and track.kind == rtc.TrackKind.KIND_AUDIO and participant.identity == phone_number:
                            recorder.attach_to_track(track)
                            recording_started = True
                            logger.info(f"Recording started: {recording_path}")
                            # Remove the event listener after recording starts
                            ctx.room.off("track_subscribed", on_track_subscribed)
                    
                    ctx.room.on("track_subscribed", on_track_subscribed)
        
        ctx.room.on("participant_connected", on_participant_connected)
        
        # Dial the user
        await ctx.api.sip.create_sip_participant(
            api.CreateSIPParticipantRequest(
                room_name=ctx.room.name,
                sip_trunk_id=outbound_trunk_id,
                sip_call_to=phone_number,
                participant_identity=phone_number,
                wait_until_answered=True,
            )
        )
        
        # Wait for session start and participant join
        await session_task
        logger.info(f"Agent session started for {phone_number}")
        
        # Wait for call to end (using room disconnection)
        disconnected = asyncio.Event()
        
        def on_disconnect():
            disconnected.set()
            ctx.room.off("disconnected", on_disconnect)
            
        ctx.room.on("disconnected", on_disconnect)
        
        try:
            await disconnected.wait()
            logger.info(f"Room disconnected: {ctx.room.name}")
        except asyncio.CancelledError:
            pass
            
    except api.TwirpError as e:
        logger.error(f"SIP error: {e}")
    finally:
        # Stop recording before finalizing
        if 'recorder' in locals():
            try:
                await recorder.stop()
            except Exception as e:
                logger.error(f"Error stopping recorder: {e}")
        
        # Ensure call is finalized
        if agent.call_outcome == "unresolved":
            await agent.finalize_call("system_error")
    ctx.shutdown()


if __name__ == "__main__":
    # Run automated testing if TEST_MODE is set
    if os.getenv("TEST_MODE") == "1":
        asyncio.run(automated_testing_entry())
    else:
        cli.run_app(
            WorkerOptions(
                entrypoint_fnc=agent_entrypoint,
                agent_name="debt-collector",
            )
        )