# Transformer based dependency parser

This is a modification of the transformer architecture in an attempt to create a very low vocabulary universal dependency parser. Instead of the traditional approach of relying on large amounts of word embeddings, it attempts to combine part of speech tag embeddings with a small number of typological language features. The input from the language features is simply added as a separate "language embedding" to the inputs of the transformer blocks in the encoder. The outputs are shift-reduce parser actions and dependency relationships if the parser action has a dependency label.

The training data (not currently included in the repo) was created from the [Universal Dependencies Treebank](https://universaldependencies.org/) dataset where I took a subset of the trees that were easily alignable with data from [WALS](https://wals.info/). The training outputs were generated from this subset of trees by using a rule based parser (included in the repo) that could use the actions: 
1. Shift - Move first element of buffer to the top of the stack.
2. Left - Draw left arc
3. Right - Draw right arc
4. Insert - Take the top element of the stack and insert it after the first element in the buffer in order to salvage crossing branches present in the data.

The language features from WALS were chosen based off an educated guess as to what would most influence word order:
1. word order -
parameter id 81A, 
SVO 81A-2, 
VSO 81A-3, 
SOV 81A-1 

2. demonstrative order - 
parameter id 88A, 
demonstrative-noun 88A-1,
noun-demonstrative 88A-2,
demonstrative-prefix 88A-3,
demonstrative-suffix 88A-4,
demonstrative-before-and-after-noun 88A-5

3. numeral order -
parameter id 89A,
numeral-noun, 
noun-numeral,
noun-numeral-no-dominant-order

4. position-of-question-particles -
parameter id 92A,
initial,
final,
second-pos,
no-question-particle

5. position of interrogative phrases in content questions -
parameter id 93A,
initial-interrogative-phrase,
not-initial-interrogative-phrase,
mixed,
unspecified

The original Transformer implementation mainly follows the same structure as this [blogpost (The Annotated Transformer)](https://nlp.seas.harvard.edu/2018/04/03/attention.html)

# Current state

So far only low accuracy has been achieved. I think this is mainly due to separating the output and the relatively low amount of training data. In the future, I plan to combine the two outputs into combination values. More effort can be made in selecting the best language features, and coming up with a better way to incorporate them into the network. I was conservative in aligning UD treebank dataset language names and WALS data, leading to me not using the majority of the UD treebank data available.
