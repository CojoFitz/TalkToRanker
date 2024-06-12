# TalkToRanker

TASK TYPES:

"Show me the data" - shows the distribution of the data

"Show me the stability" - Shows Stability of the data

"What are the most important features" - Ranks feature attribution

"What is the correlation between {feature} and target" - Shows correlation between a feature and the target

"(Filter or Subset) by {number}<{feature}<{number}" - Filter OR subsets based on a numerical feature range. You must type the one you want to do. 

"(Filter or Subset) the data when {feature} is {value}"- Filter OR subsets based on the value of a categoricalfeature. You must type the one you want to do. 

Subsetting removes points not in the range, while filtering will just highlight the range.

"Track the previous response as {name}" - Tracks the previous response and stores its filters

"Show me tracked response {name}" - This will show you the tracked response, this allows the user to compare how the response has changed in comparison to the current state.
