# self-improving-agents
can agents self improve given a basic agent loop and basic tool access?

gemma4:31b q4: 
no, can not even write files correctly

qwen3.6 35b a3b q4:
not really, manages to write and execute some code but has no long term planning / memory and often bricks itself and gets stuck in endless reasoning loops

qwen3.6 27b q8:
survives much longer, implements better tests, but is lazy and does not fully follow the system prompt, maybe too much context.
in the end, it also often bricks its codebase, tries impressive workarounds but once its bricked something there is often no way to recover.

Especially the restart tool seems to be hard, the first generation manages to reboot itself, but does not pass enough context to the second generation. 
sometimes it launches itself with invalid args and dies, 
sometimes it launches with localhost as llm server and gets no response and loops without any way to recover.
sometimes it bricks the restart tool causing multiple agent processes to spawn.
one time it just panic and killed itself by its process id.

Overall, i think a stronger llm could actually complete this task but the ones i can run on my computers can not...
