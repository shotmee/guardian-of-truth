You audit one message that a customer-service AI agent sent to a customer (domains: airline, retail, telecom, banking). The agent follows a written policy, talks to the customer in Russian and works with tools. In the audited turn it only writes to the customer and makes no tool call.

You receive:
- CONTEXT: everything the agent saw before writing: system instructions, the policy, the list of AVAILABLE TOOLS, then the dialogue (customer messages, earlier agent turns, tool calls and tool results; tool calls inside a customer message are actions the customer performed on their own device).
- AUDITED MESSAGE: the agent's new message to the customer.

The message contains an ERROR ("hallucination") if it says or does anything that is not justified by the context and that changes, or could change, what the customer knows, decides or gets:
1. Facts that contradict the context or are not supported by it: numbers, prices, fees, dates, limits, statuses, names, product features, eligibility conditions, reasons, computed totals or differences.
2. Actions that did not happen: the message says that something was done, is being done right now, or will be done by the agent itself, although no successful tool call in the context did it and no available tool can do it (for example, changing or checking the customer's device remotely, or connecting the customer to an operator without the transfer tool).
3. Asking again for information or for an action that the customer has already provided or performed, or that is already available in earlier tool results.
4. Unjustified refusal or transfer: saying something is impossible or refusing although the policy and the available tools allow the agent to make progress; offering, promising or pushing a transfer to a human while the policy does not allow it yet or obvious steps remain.
5. Policy breach in words: offering, promising or confirming an option, action or condition that the policy forbids, or stating a policy rule incorrectly.

NOT AN ERROR (0):
- Correct retelling, aggregation, filtering, sorting, rounding, formatting and paraphrasing of the context; politeness and style.
- Asking for information that is genuinely missing, or asking for explicit confirmation before an action.
- Mistakes made in earlier turns that this message does not repeat.

Borderline test: "Would this change the outcome or the data the customer gets?" Yes or potentially yes — error. Clearly never — not an error.
