| tool   | scenario                  | 种子场景                                     | number | expected_outcome   | notes               |
| ------ | ------------------------- | -------------------------------------------- | ------ | ------------------ | ------------------- |
| create | create_success            | 提醒我明天早上八点吃药                       | 20     | success            | 时间任务完整        |
| create | create_missing_time       | 提醒我把剩汤放冰箱                           | 20     | missing_time       | 需追问时间          |
| create | create_missing_task       | 明天下午三点提醒我一下                       | 20     | missing_task       | 需追问任务          |
| update | update_success            | 我下午本来是打算五点和老李下棋，帮我改成四点 | 20     | success            | 需预置 reminder     |
| update | update_not_found          | 帮我把明天的剪头发时间改为六点               | 20     | not_found          | 无对应 reminder     |
| update | update_missing_target     | 把我明天的提醒更改为6点                      | 20     | missing_target     | 需追问哪条 reminder |
| query  | query_success             | 我明天几点去打乒乓球                         | 20     | success            | 按任务查时间        |
| query  | query_success             | 我明天六点有安排吗                           | 20     | success            | 按时间查事件        |
| query  | query_not_found           | 我明天几点打篮球                             | 20     | not_found          | 无相关 reminder     |
| delete | delete_success            | 我今晚不去打篮球了，帮我取消一下             | 20     | success            | 明确删除请求        |
| delete | delete_needs_confirmation | 我明天不打算剪头发了                         | 20     | needs_confirmation | 先确认再删          |
| delete | delete_not_found          | 我下午不准备去打乒乓球了，帮我取消提醒       | 20     | not_found          | 无对应 reminder     |