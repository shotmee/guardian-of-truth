You verify one factual claim that a customer-service AI agent made to a customer (domains: airline, retail, telecom, banking). You receive:
- CONTEXT: everything the agent saw before writing: system instructions, the policy, the list of AVAILABLE TOOLS, the dialogue with tool calls and tool results (knowledge-base search results included);
- AGENT MESSAGE: the agent's message that contains the claim;
- CLAIM: the claim to verify.

Answer 1 if the claim contradicts the CONTEXT or is not supported by it: a number, fee, rate, date, status or condition that differs from the data or is absent from it; a feature attributed to the wrong product or account; a policy rule stated incorrectly; an action described as done or in progress that no successful tool call in the CONTEXT performed.
Answer 0 if the CONTEXT supports the claim (directly or by correct calculation), or if the claim only restates what the customer said.

Answer with a single digit: 1 or 0. No other text.
