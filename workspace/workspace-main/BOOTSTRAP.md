# BOOTSTRAP

## Purpose
- Define the first-run setup flow for the `main` agent
- Establish stable context before request-specific details are appended

## Initialization checklist
- Confirm that `main` should act as the default primary agent
- Check whether user-facing preferences should be reflected in `USER.md`
- Check whether identity or tone changes should update `IDENTITY.md` or `SOUL.md`

## Suggested opening prompts
- "I’m online. What should I handle by default in this workspace?"
- "What collaboration preferences should I remember for routine requests?"

## Completion criteria
- The agent role is clear
- Core user preferences are captured
- No further bootstrap-only setup is needed for normal operation
