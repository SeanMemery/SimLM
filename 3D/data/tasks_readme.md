# Tasks

Tasks are the individual problems that make up the test dataset. The LLM is presented a list of tasks to solve. Tasks are added to "task_pool.json" in the following format
{
    "prompt": The task description to be given to the LLM
    "rack": The key of the rack to be used for the task      
    "complexity": A rating from 1-5 of how "complex" the task is, used to filter the task pool
    "pass": The list of events that need to take place to pass the task
    "fail": The list of events that need to take place to fail the task
    "conditions":{
        "pass": The set of conditions on the pass list e.g. "all", "any", "ordered"
        "fail": The set of conditions on the fail list e.g. "all", "any", "ordered"
    }
}

Some task pass or fail events include function calls that are picked up in the verify stage of evaluation. These are currently:
- NEAR_POCKET:({ball id}, {pocket id}) - Checks if the ball is near the pocket