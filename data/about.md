
**Please note that this app is intended for testing purposes only.**

## Paremeters

#### Models
* ChatGPT 3.5 Turbo 16k: it can accept 16k tokens or more than 10k words; and it is faster, can add more information and requirement; cheaper

* ChatGPT 4: "more intelligent (?)", can only accept 8k tokens; slow and may have timeout errors; more expensive

* CHatGPT Turbo 128K: not yet available



#### Audience and Style:
* Delegates/Heads of states: Diplomatic, formal, strategic, respectful, authoritative, policy-oriented, persuasive, factual, concise, collaborative,
* Think tanks:  Diplomatic, Strategic, Authoritative, Analytical, Persuasive, Forward-thinking, Inclusive, Policy-focused, Insightful, Collaborative',
* Academics: Scholarly, Analytical, Informed, Thought-provoking, Collaborative, Insightful, Respectful, Comprehensive, Evidence-based, Innovative',
* Students: Inspirational, engaging, informative, motivational, relatable, empathetic, uplifting, visionary, accessible, encouraging',





## Technonogy


### Text Generation with Retrieval Augmented Generation (RAG)

Retrieval Augmented Generation (RAG) is an innovative approach in natural language processing (NLP) and artificial intelligence (AI) that combines the strengths of retrieval-based and generative models to enhance text generation. In this method, a retrieval-based model first identifies and retrieves relevant information from a large dataset, based on the given input, such as a question or prompt. This model is adept at sourcing existing, pertinent content but does not generate new text. The retrieved information is then supplied to a generative model like GPT (Generative Pre-trained Transformer), known for its ability to create new, contextually relevant content. By utilizing the retrieved data alongside the initial input, the generative model crafts responses that are not only contextually appropriate but also rich in factual details and specificity. This synergy allows RAG to produce outputs that are both informative and nuanced, making it highly valuable in applications where precise, detailed, and context-aware responses are essential, such as advanced chatbots, question-answering systems, and other AI applications. RAG represents a significant leap in AI's capability to generate responses that are closer to human-like accuracy and relevance, enhancing the quality and reliability of AI-driven interactions.





### Semantic Search using Text Embeddings


The document search engine incorporates a new search feature that utilizes text embeddings from OpenAI's large language model, 
providing more accurate and relevant search results. Here's an explanation of what this means and how it differs from traditional search methods.

##### What are text embeddings?

Text embeddings can be thought of as a way to convert complex language into a simpler, mathematical representation that computers 
can understand more easily. By representing words, phrases, and paragraphs as points in a high-dimensional space, text embeddings 
enable our search engine to process and interpret the meaning of the text more effectively.

##### How is this search different from traditional text matching search?

Traditional search engines often rely on matching keywords or phrases from your search query to those found in documents. While this 
method has its merits, it can sometimes miss the true meaning or context behind the words being used.

This new search feature, powered by text embeddings from OpenAI's large language model, takes a more sophisticated approach. By converting 
both your search query and the documents into embeddings, we can measure the similarity between them in the high-dimensional space. 
This allows our search engine to better understand the meaning and context of the words, phrases, and sentences, providing more relevant 
and accurate results that better align with your intentions.

##### How should you construct your query?

Be clear and specific when crafting your query. There's no need to worry about whether the words or phrases will exactly match the text 
you want to find. The search engine will focus on understanding the meaning behind your query and deliver relevant results accordingly.

You can use English, French, Spanish, Arabic, German and other languages.         

                            |                                          |

## Change log
* V0.2 (20240111): Rearrange menu; refined perameter selection; new prompt text; 
* V0.1 (20231220): Initial version