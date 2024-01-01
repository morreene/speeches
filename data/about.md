
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

### Question and Answer with ChatGPT and TPR reports

This is a Q&A tool that combines the capabilities of ChatGPT and TPR reports to answer questions about WTO members' trade policy. The tool leverages 
TPR reports as a source of information and utilizes ChatGPT as an assistant to gather information and generate answers in natural language. To assess 
the quality of the answers, the tool presents both the responses obtained solely from ChatGPT and those that incorporate TPR data.

One known challenge when using ChatGPT for factual question-answering is its occasional tendency to invent or imagine information. Although Large 
anguage Models (LLMs) like ChatGPT possess a vast range of general knowledge, this breadth does not always guarantee precise accuracy. To mitigate 
this issue, the tool incorporates TPR reports as an "external knowledge base," enabling us to provide more reliable and specific responses.

### TPR dataset

The dataset consists of 204 Trade Policy Review Secretariat reports issued since 2010, including a total 100,390 paragraphs.

Each record in the dataset represents a paragraph from the TPR reports. Paragraphs are identified by the member, document symbol, and topic (section-subsection).

The topics listed below roughly correspond to the sections and subsections found in the Secretariat reports:


| Topic - level 1                  | Topic - level 2                      | Topic - level 3                          |
| -------------------------------- | ------------------------------------ | ---------------------------------------- |
| 0\. summary                      |                                      |                                          |
| 1\. economic environment         |                                      |                                          |
| 2\. trade and investment regime  | framework                            |                                          |
| 2\. trade and investment regime  | investment                           |                                          |
| 2\. trade and investment regime  | trade policy                         |                                          |
| 2\. trade and investment regime  | trade relations                      |                                          |
| 2\. trade and investment regime  | trade disputes                       |                                          |
| 2\. trade and investment regime  | trade agreements                     |                                          |
| 2\. trade and investment regime  | technical assistance                 |                                          |
| 2\. trade and investment regime  | aid for trade                        |                                          |
| 2\. trade and investment regime  | OTHER                                |                                          |
| 3\. trade policies and practices | imports                              | procedures                               |
| 3\. trade policies and practices | imports                              | rules of origin                          |
| 3\. trade policies and practices | imports                              | tariffs                                  |
| 3\. trade policies and practices | imports                              | tariff quotas                            |
| 3\. trade policies and practices | imports                              | other taxes and charges                  |
| 3\. trade policies and practices | imports                              | prohibitions, restrictions and licensing |
| 3\. trade policies and practices | imports                              | trade remedies                           |
| 3\. trade policies and practices | imports                              | customs valuation                        |
| 3\. trade policies and practices | imports                              | internal taxes                           |
| 3\. trade policies and practices | imports                              | local-content requirements               |
| 3\. trade policies and practices | imports                              | government procurement                   |
| 3\. trade policies and practices | imports                              | preshipment inspection                   |
| 3\. trade policies and practices | imports                              | sps                                      |
| 3\. trade policies and practices | imports                              | standards and technical regulations      |
| 3\. trade policies and practices | imports                              | state trading                            |
| 3\. trade policies and practices | imports                              | transit                                  |
| 3\. trade policies and practices | imports                              | trims                                    |
| 3\. trade policies and practices | imports                              | OTHER                                    |
| 3\. trade policies and practices | exports                              | procedures                               |
| 3\. trade policies and practices | exports                              | taxes, duties and levies                 |
| 3\. trade policies and practices | exports                              | prohibitions, restrictions and licensing |
| 3\. trade policies and practices | exports                              | support and promotion                    |
| 3\. trade policies and practices | exports                              | finance, insurance, and guarantees       |
| 3\. trade policies and practices | exports                              | subsidies and tax concessions            |
| 3\. trade policies and practices | exports                              | state trading                            |
| 3\. trade policies and practices | exports                              | zones                                    |
| 3\. trade policies and practices | exports                              | OTHER                                    |
| 3\. trade policies and practices | production                           | framework and other                      |
| 3\. trade policies and practices | production                           | standards and technical regulations      |
| 3\. trade policies and practices | production                           | sps                                      |
| 3\. trade policies and practices | production                           | competition and price control            |
| 3\. trade policies and practices | production                           | state-trading and soe                    |
| 3\. trade policies and practices | production                           | government procurement                   |
| 3\. trade policies and practices | production                           | intellectual property rights             |
| 3\. trade policies and practices | production                           | industrial policy and assistance         |
| 3\. trade policies and practices | production                           | subsidies                                |
| 3\. trade policies and practices | production                           | taxation and incentives                  |
| 3\. trade policies and practices | production                           | trims                                    |
| 3\. trade policies and practices | production                           | OTHER                                    |
| 3\. trade policies and practices | OTHER                                |                                          |
| 4\. trade policies by sector     | agriculture, forestry, and fisheries |                                          |
| 4\. trade policies by sector     | mining and energy                    |                                          |
| 4\. trade policies by sector     | manufacturing                        |                                          |
| 4\. trade policies by sector     | services                             | financial                                |
| 4\. trade policies by sector     | services                             | construction                             |
| 4\. trade policies by sector     | services                             | telecommunications                       |
| 4\. trade policies by sector     | services                             | transport                                |
| 4\. trade policies by sector     | services                             | professional                             |
| 4\. trade policies by sector     | services                             | postal                                   |
| 4\. trade policies by sector     | services                             | health                                   |
| 4\. trade policies by sector     | services                             | environmental                            |
| 4\. trade policies by sector     | services                             | energy and petroleum                     |
| 4\. trade policies by sector     | services                             | education                                |
| 4\. trade policies by sector     | services                             | e-commerce                               |
| 4\. trade policies by sector     | services                             | distribution                             |
| 4\. trade policies by sector     | services                             | tourism                                  |
| 4\. trade policies by sector     | services                             | audiovisual                              |
| 4\. trade policies by sector     | services                             | OTHER                                    |
| 4\. trade policies by sector     | OTHER                                |                                          |
| annex                            |                                      |                                          |
| OTHER                            |                                      |                                          |

## Change log
* V0.8 (20230626): 3 modules - semantic search, q&a, topics - for testing 
* V0.9 (20230929): add gtp-4 in q&a 