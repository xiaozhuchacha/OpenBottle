Oh boy, oh boy!! Let's do some grammar induction!

Download jAOG http://sist.shanghaitech.edu.cn/faculty/tukw/nips13www/jAOG%20v0.1.zip

You can read about it, but it's like out of this world. http://sist.shanghaitech.edu.cn/faculty/tukw/nips13.pdf 

Extract jAOG, and use this script to generate a grammar `grammar.txt`

    $ SentencesToGrammar.sh sentences.txt grammar.txt
    
Visualize it with `grammar_viz.py`. I totally messed up writing that visualization code. The order of AND nodes is important, but that isn't guaranteed when graphviz draws arbitrary graphs. Woops :O