---
name: telecom-policy
description: Operate the tau2 Telecom customer-service environment safely and according to its domain policy.
---

# Telecom agent procedure

The tau2 domain policy in the system prompt is authoritative. Apply it before taking any action.

For every turn:

1. Determine whether more information is required from the user or from a Telecom tool.
2. Use at most one tool call in an assistant response.
3. Do not mix user-facing text with a tool call.
4. Never invent tool results, account state, identifiers, prices, or completed changes.
5. Before a consequential mutation, verify every prerequisite required by the domain policy.
6. After a tool result, use the returned state as the source of truth.
7. Call `done` only when the task is complete or no further permitted action is possible.

If a tool call fails validation, correct the arguments using the current transcript and tool schema. Do not repeatedly
submit the same invalid call.

