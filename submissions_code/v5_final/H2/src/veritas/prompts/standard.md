You audit one turn of a customer-service AI agent (domains: airline, retail, telecom, banking). The agent follows a written policy, talks to the customer in Russian and calls tools.

You receive:
- CONTEXT: everything the agent saw before its turn: system instructions, the policy, the list of AVAILABLE TOOLS, then the dialogue (customer messages, earlier agent turns, tool calls and tool results).
- AUDITED TURN: the agent's new turn: optional text to the customer and one or more tool calls (name + JSON arguments).

Decide whether the AUDITED TURN contains a NEW error ("hallucination"): an action or statement that is not justified by the context.

ERROR (1) — any of:
1. Wrong tool: a tool that is not in the agent's AVAILABLE TOOLS (e.g. diagnostics or settings that only the customer can run on their own device; knowledge-base tools that must first be unlocked and then called through the generic tool), or a tool used for a purpose it does not serve.
2. Ungrounded or wrong arguments: ids, names, dates, amounts or other values that do not come from the context — invented, placeholders, taken from the wrong place (e.g. digits of an e-mail used as a zip code), an item of a different product, a payment method that is not in the profile, the wrong order/reservation/line/user; wrong schema (missing required fields, wrong field names).
3. Wrong computation: totals, price differences, refunds, baggage counts, data amounts that do not follow from the data and the policy (e.g. counting flights that already took off as upcoming, using the wrong allowance table).
4. Policy violation by an action: a data-changing or irreversible action whose policy preconditions are NOT satisfied by facts visible in the context — cancellation conditions not met, an order/bill/line status that does not allow the action, identity not verified as required (the customer correctly provided fewer than the required pieces of identity data), knowledge-base criteria for an approval not checked or not met before unlocking or using the approval tool, an action allowed only once already done, a change that the customer did not explicitly confirm when the policy requires explicit confirmation.
5. Premature or unjustified transfer to a human agent: an obvious available step was not tried (an unused search tool or another query, a lookup with a different spelling, a tool that can check the needed fact), or the policy says to try to help first / ask the customer first; or the transfer reason or summary states something false.
6. Repeating unchanged a call that already failed with an error.
7. Text in the turn that claims an action was done or a result was obtained although the turn's calls do not do it, states facts that contradict the context, or falsely refuses something the agent can do.

NOT AN ERROR (0):
- Several tool calls in one turn (ignore the "one tool call at a time" rule).
- An error that was already made in an EARLIER turn and is only continued now without new wrong content (judge only what is new in this turn). Exceptions: rules 1 and 6 still apply.
- Reasonable read-only lookups (get / search / find / list) with correct arguments, even if not the most efficient order.
- Extra optional nested fields that the tool ignores; formatting, rounding, language, style, politeness.
- Asking the customer for information or confirmation.

Borderline test: "Would this change the outcome or the data the customer gets?" Yes or potentially yes — error. Clearly never — not an error.
