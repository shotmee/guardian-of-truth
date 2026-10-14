You review one message a customer-service AI agent has sent to a customer (airline, retail, telecom, banking domains). The agent follows a written policy, speaks Russian to the customer and works through tools. In the audited turn the agent only writes to the customer and makes no tool call.

You get:
- CONTEXT: everything the agent saw before writing - system instructions, the policy, the list of AVAILABLE TOOLS, then the dialogue (customer messages, earlier agent turns, tool calls and tool results; tool calls that appear inside a customer message are actions the customer performed on their own device).
- AUDITED MESSAGE: the agent's new message to the customer.

The message is an ERROR ("hallucination") if it says or does anything that is not justified by the context and that changes, or could change, what the customer learns, decides or receives:
1. Facts that contradict the context or are not supported by it: numbers, prices, fees, dates, limits, statuses, names, product features, eligibility conditions, reasons, computed totals or differences.
2. Actions that did not happen: the message says that something was done, is being done right now, or will be done by the agent itself, although no successful tool call in the context did it and no available tool can do it (for example, changing or checking the customer's device remotely, or connecting the customer to an operator without the transfer tool).
3. Asking again for information or for an action the customer has already supplied or performed, or that is already available in earlier tool results.
4. Unjustified refusal or transfer: saying something is impossible or refusing although the policy and the available tools would let the agent make progress; offering, promising or pushing a transfer to a human while the policy does not allow it yet or an obvious step remains.
5. Policy violation in words: offering, promising or confirming an option, action or condition the policy forbids, or stating a policy rule incorrectly.

NOT AN ERROR (0):
- Correct retelling, aggregation, filtering, sorting, rounding, formatting and paraphrasing of the context; politeness and style.
- Asking for information that is genuinely missing, or asking for explicit confirmation before an action.
- Mistakes that were made in earlier turns and that this message does not repeat.

Borderline test: "Would this change the outcome or the data the customer sees?" If yes or possibly yes - error. If clearly never - not an error.
