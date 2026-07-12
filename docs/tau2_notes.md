# tau2 Notes

- `Task.user_scenario` is for the user simulator, not the trainable agent.
- `Task.evaluation_criteria` is for evaluation, not for the trainable agent prompt.
- The trainable agent receives the domain policy, assistant tool schemas, and
  the current conversation observation.
- Long observations may be shortened by deterministic, episode-local context
  compression. No hidden task answers or cross-task state may enter the prompt.
- The official tau2 evaluator remains the authority for outcome reward. Tool
  success alone does not imply task success.
