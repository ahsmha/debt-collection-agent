### Demo
Debt collector agent - https://youtu.be/HjUfjQC64us

Testing - https://youtu.be/Jha6YKNh0TE

## Dev Setup

Clone the repository and install dependencies to a virtual environment:

```shell
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python agent.py download-files
```

Set up the environment by copying `.env.example` to `.env.local` and filling in the required values:

Run the agent:

```shell
python3 agent.py dev
```

Now, your worker is running, and waiting for dispatches in order to make outbound calls.

### Making a call

You can dispatch an agent to make a call by using the `lk` CLI:

```shell
lk dispatch create \
  --new-room \
  --agent-name outbound-caller \
  --metadata '{"phone_number": "+12179926018", "transfer_to": "+918977016552}'
```
