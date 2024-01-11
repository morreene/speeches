
**Please note that this app is intended for testing purposes only.**

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
* V0.8 (20230626): 3 modules - semantic search, q&a, topics - for testing 
* V0.9 (20230929): add gtp-4 in q&a 