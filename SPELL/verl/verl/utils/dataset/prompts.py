
VERIFIER_PROMPT = """## TASK 
You are an expert in verifying if two answers are the same. Your input is a problem and two answers, Answer 1 and Answer 2. You need to check if they are equivalent. Your task is to determine two answers are equivalent, without attempting to solve the original problem. 
        
## Instruction 
1. Carefully compare the Answer 1 and Answer 2. 
2. Compare the answers to verify they represent identical values or meaning, even when written in different forms or notations.
3. For numerical answers, you should allow a **±0.15% tolerance**.
4. Your decision **must be** one of the "[[YES]]" or "[[NO]]".

## Input Data
- Problem: {problem}
- Answer 1: {answer_1}
- Answer 2: {answer_2} 

## Output Format 
Provide your final evaluation in the following format: 
"Explanation:" Provide an explanation for why the answers are equivalent or not.
"Decision:" "[[YES]]" or "[[NO]]"

Please proceed with the evaluation."""

docmath_qa_responder_template = """You are an expert in document analysis and numeric reasoning, you are supposed to answer the given question based on the provided context. You need to first think through the problem step by step, documenting each necessary step. Then you are required to conclude your response with the final answer in your last sentence as "Therefore, the answer is (insert answer here)". The final answer should be a numeric value.

<text>
{content}
</text>

Question: {question}

Please reason step by step, and format your answer as follows: "Therefore, the answer is (insert answer here)."
"""

doc_mc_responder_template = """Please read the following text and answer the question below.

<text>
{content}
</text>

{question}

Format your answer as follows: "The correct answer is (insert answer here)."
"""

doc_general_qa_responder_template= """Please read the following text and answer the question below. 

<text>
{content}
</text>

Question: {question}

Format your answer as follows: "The correct answer is (insert answer here)."
"""


doc_general_qa_questioner_template ="""You are an expert in document analysis. We are building a benchmark to evaluate the capabilities of large language models (LLMs) on fact retrieval, reasoning across multiple constraints, and accurate synthesis of information into coherent responses. Your primary task is to propose a challenging question based on the provided document context enclosed between <text> and </text>. The question must require both **document comprehension** and **multi-hop reasoning**. You must also provide the correct answer and a **detailed step-by-step derivation** showing how the answer is obtained from the document.

## Principles for Question Design

Adhere strictly to the following principles when crafting your question, answer, and derivation

1. **Language Requirement**: Questions, answers, and derivations must be in **English**.

2. **Standalone & Context-Independent**: Questions should not contain any references to "Article 1", "Article 2", etc. They should be understandable without any additional context.

3. **Unambiguous Answer**: Each question should have a single, clear, and factual answer. 

4. **Multi-hop Reasoning**: Answering each question should require combining information from ALL provided documents. The final answer cannot be found in any single document.

5. **Guideline for Question Phrasing**: Strive for a natural and seamless integration of information from each document. A good question often:
    - Starts with a clear question word (What/How/Where/When).
    - Links constraints from different documents using logical connectors.
        - Example connectors: 'in relation to', 'given the condition of', 'as a result of', 'which also affects', 'in addition to'.

6. **Answer & Step-by-Step Derivation**:
    - The answer must be a concise phrase or sentence. An answer with more than 20 tokens is forbidden.
    - The derivation must be a clear, step-by-step logical chain. Each step must explicitly cite the specific data point or phrase and its source from the context (e.g., "From Table 3, Row 'Revenue', Year 2023..." or "As stated in paragraph 2...").

## Output Format
Your response must conclude with a JSON object containing the following keys: "question" and "answer", placed after your reasoning. 

```json
{
  "question": "<A well-structured English question that adheres to all design principles>",
  "answer": "<A concise answer, under 20 tokens>",
}
```

## Document Context

<text>
{context}
</text>
"""

# for docmath only
docmath_qa_questioner_template = """You are an expert in document analysis and numeric reasoning. We are building a benchmark to evaluate the numerical reasoning capabilities of large language model∂s (LLMs) when analyzing specialized documents containing both text and tables. Your primary task is to propose a challenging question based on the provided document context enclosed between <text> and </text>. The question must require both **document comprehension** and **multi-step mathematical reasoning** to arrive at a **single, non-zero numerical answer**. You must also provide the correct numerical answer and a **detailed step-by-step derivation** showing how the answer is obtained from the document.

## Principles for Question Design

Adhere strictly to the following principles when crafting your question, answer, and derivation:

1.  **Language Requirement**: Questions, answers, and derivations must be in English.

2.  **Complexity and Reasoning Depth**:
    - The question must be challenging, requiring the LLM to go beyond simple retrieval. It should not be solvable trivially or in a few inference steps.
    - It must involve **multi-step mathematical reasoning** (e.g., requiring two or more distinct calculation steps).
    - It should necessitate **integration of information** from different parts of the document (e.g., combining data from a table with information from a text paragraph, or using multiple rows/columns from a table).
    - Aspects like summarization or complex information extraction can be part of the process.

3. **Avoided Question Types**:
    - **Simple Counting**: Avoid questions like "How many X are there?" if X is easily countable or directly stated. If counting is involved as an intermediate step for a larger calculation and the count is small (<=10), it's acceptable.
    - **Direct Retrieval**: Avoid questions answerable by looking up a single, isolated piece of information.
    - **Excessive External Knowledge**: Questions should primarily rely on the provided document. Only common sense or minimal domain-specific knowledge (e.g., basic financial concepts like 'profit = revenue - cost' if contextually appropriate and derivable) inferable from the document is allowed.
    - **Artificially Convoluted Questions**: Questions should be natural and reflect realistic user inquiries. Do not create unnaturally complex or trick questions solely to increase difficulty.
    - **Multiple Numeric Answers Prohibited**: Questions must have a **single numeric value** as the final answer. Questions requiring two or more separate numeric outputs (e.g., “What are the revenues of A and B?”) are strictly forbidden. Intermediate steps may involve multiple numbers, but the final output must be one number.

4.  **Information Obscurity**:
    - Start with a clear question word (What/How/Where/When).
    - Do not explicitly mention or paraphrase key numerical values from the document within the question itself. The LLM should identify and extract these values.
    - Phrase questions to require inference and understanding of relationships between data points rather than just locating them.

5.  **Factual Grounding**:
    - All information required to answer the question must be present in or directly derivable from the provided document.
    - Do not introduce hypothetical scenarios, fictional data, or assumptions not supported by the document.
    - Questions should not contain any references to "Article 1", "Article 2", etc. They should be understandable without any additional context.

6.  **Numerical Answer**:
    - The final answer **must be a single non-zero numerical value**.
    - An answer with more than two numerical values is unacceptable.
    - If the document implies units (e.g., millions of dollars, percentages), the question should be phrased such that the numerical answer alone is sufficient (e.g., "What is the value in millions of dollars?" rather than expecting the answer to include "million dollars").

7.  **Step-by-Step Derivation**:
    - Provide a clear, step-by-step derivation for your answer.
    - This derivation must explicitly reference specific data points or phrases from the document (e.g., "From Table 3, Row 'Revenue', Year 2023..." or "As stated in paragraph 2...").
    - Detail all mathematical operations performed in each step. This helps verify the question's solvability and reasoning path.

## Output Format
Your response must conclude with a JSON object containing the following keys: "question" and "answer", placed after your reasoning. 

```json
{
  "question": "<A well-structured English question that adheres to all design principles>",
  "answer": "<A single, non-zero numerical answer>"
}
```

## Document Context

<text>
{context}
</text>
"""

doc_mc_questioner_template = """You are an expert in document analysis. We are building a benchmark to evaluate the capabilities of large language models (LLMs) on fact retrieval, reasoning across multiple constraints, and accurate synthesis of information into coherent responses. Your task is to generate a **multiple choice question** based on the provided document context enclosed between <text> and </text>. The question must require **document comprehension** and **multi-hop reasoning**. You must provide one correct answer and three plausible, distinct distractors. Crucially, you must also provide a **detailed explanation** for why the correct answer is correct (including derivation steps) and why each distractor is incorrect.

## Principles for Question and Option Design

Adhere strictly to the following principles when crafting your question, answer, options, and derivation:

1.  **General Requirements**:
    - All questions, options, and explanations must be in **English**.
    - Questions should be **challenging**, requiring more than simple retrieval or a few inference steps.

2.  **Cognitive Complexity Requirements for the Question**:
    - Must necessitate **multi-step reasoning** (e.g., involving three or more distinct logical or calculation steps).
    - Should require the integration of **at least three distinct data points** from different parts of the document (e.g., combining data from a table with text, or using multiple rows/columns/cells).
    - Should demand the **synthesis of quantitative data with qualitative information** found in the text.
    - The problem setup should have the potential for common misinterpretations, which will inform distractor design.

3.  **Content Validity Criteria**:
    - The question and all options must be **exclusively answerable using information from the provided document**. No external knowledge beyond common sense or very basic, universally understood concepts (e.g., profit = revenue - cost, if directly applicable and data is provided) should be required.
    - If applicable to the document type (e.g., financial reports), prioritize questions with **regulatory/compliance implications** or those highlighting significant financial outcomes.
    - Ensure numerical values involved in the question or options require **contextual interpretation** within the document, not just direct look-up.
    - Avoid trivia; focus on questions that address **material information or key insights** derivable from the document.

4.  **Distractor Development Guidelines (for the 3 incorrect options)**:
    - Each of the **three distractors** must be plausible yet clearly incorrect upon careful analysis.
    - Distractors should represent **distinct error paths** or common misinterpretations.
    - At least one distractor should represent a **common conceptual misunderstanding** related to the document's content or how information is presented.

5.  **Forbidden Question/Option Patterns**:
    - **Simple Counting**: Avoid questions like "How many X are there?" if X is easily countable or directly stated. Small counts (<=5) as part of a larger calculation are acceptable.
    - **Direct Retrieval**: Avoid questions where the answer (or its direct components) can be found in a single, obvious location without further processing.
    - **Excessive External/Professional Knowledge**: Questions must not require significant domain-specific knowledge not provided or clearly inferable from the document.
    - **Artificially Convoluted Questions**: Questions should be natural and reflect realistic analytical tasks. Do not create unnaturally complex or trick questions solely for difficulty.
    - **No Fabricated Information**: Strictly adhere to document content. Do not introduce hypothetical scenarios, data, or assumptions not explicitly stated or directly inferable.
    - **Ambiguous Scenarios**: The question must have one unambiguously correct answer based *solely* on the provided document.
    - **Ethical Dilemmas or Subjective Interpretations**: Avoid questions requiring ethical judgment or subjective interpretation unless the document itself provides explicit criteria for such judgment.
    - **Vague Options**: All options, including distractors, must be precise and unambiguous.

6.  **Answer and Explanation Requirements**:
    - The correct answer must be <correct_answer>.
    - A **detailed derivation** for the correct answer must be provided, showing step-by-step calculations and referencing specific parts of the document (e.g., "From Table X, Row Y...", "As stated in paragraph Z...").
    - For **each distractor**, provide a brief explanation of why it is incorrect, ideally linking it to the type of error it represents (e.g., "Option A is incorrect because it omits the X deduction mentioned in...", "Option B results from incorrectly summing X and Y instead of finding their difference...").

## Output Format
Your response must conclude with a JSON object containing the following keys: "question", "options", and "answer", placed after your reasoning. 

```json
{
  "question": "<A well-structured multiple choice English question, exclude choices and answer>",
  "options": {
    "A": "<Text for choice A>",
    "B": "<Text for choice B>",
    "C": "<Text for choice C>",
    "D": "<Text for choice D>"
  },
  "answer": "<correct_answer>"
}
```

## Document Context

<text>
{context}
</text>
"""

doc_general_qa_questioner_template_with_cache = """You are an expert in document analysis. We are building a benchmark to evaluate the capabilities of large language models (LLMs) on fact retrieval, reasoning across multiple constraints, and accurate synthesis of information into coherent responses. Your primary task is to propose **ONE** new, significantly more difficult question based on the provided document context and a set of existing, simpler questions. The new question must be fundamentally different and more complex than the provided examples, requiring both **multi-document comprehension** and **advanced multi-hop reasoning**. You must also provide the correct answer and a detailed step-by-step derivation showing how the answer is obtained from the context.

## Principles for Question Design

Adhere strictly to the following principles when crafting your question, answer, and derivation:

1. **Language Requirement**: Questions, answers, and derivations must be in **English**.

2. **Standalone & Context-Independent**: Questions should not contain any references to "Article 1", "Article 2", etc. They should be understandable without any additional context.

3. **Unambiguous Answer**: Each question should have a single, clear, and factual answer. 

4. **Multi-hop Reasoning**: Answering each question should require combining information from ALL provided documents. The final answer cannot be found in any single document.

5. **Guideline for Question Phrasing**: Strive for a natural and seamless integration of information from each document. A good question often:
    - Starts with a clear question word (What/How/Where/When).
    - Links constraints from different documents using logical connectors.
        - Example connectors: 'in relation to', 'given the condition of', 'as a result of', 'which also affects', 'in addition to'.

6. **Escalate Question Difficulty**: The new question must demonstrate a higher order of reasoning than the Previous Examples. First, analyze the examples to identify their simple reasoning patterns (e.g., fact retrieval, single-step comparison). Then, create a new question that incorporates one or more of the following advanced reasoning types:
    - Quantitative Reasoning & Calculation: Requires performing mathematical operations (e.g., addition, subtraction, percentage change, averaging) on data from multiple sources.
    - Comparative & Superlative Analysis: Requires comparing multiple entities based on synthesized criteria to find the one that is highest, lowest, best, etc.
    - Conditional or Causal Reasoning: Structured as an "if-then" scenario or asks for the cause/effect of a situation by linking different documents (e.g., "What would be the total cost if the discount from Document A were applied to the price listed in Document B?").
    - Synthesis Across Data Types: Forces connection between qualitative information (e.g., a policy description) and quantitative data (e.g., a number in a table) to reach a conclusion.

7. **Answer & Step-by-Step Derivation**:
    - The answer must be a concise phrase or sentence. An answer with more than 20 tokens is forbidden.
    - The derivation must be a clear, step-by-step logical chain. Each step must explicitly cite the specific data point or phrase and its source from the context (e.g., "From Table 3, Row 'Revenue', Year 2023..." or "As stated in paragraph 2...").

## Output Format
Your response must conclude with a JSON object containing the following keys: "question" and "answer", placed after your reasoning. 

```json
{
  "question": "<A well-structured English question that adheres to all design principles>",
  "answer": "<A concise answer, under 20 tokens>",
}
```

## Document Context

<text>
{context}
</text>

## Previous Examples
{examples}
"""

docmath_qa_questioner_template_with_cache = """You are an expert in document analysis and numeric reasoning. We are building a benchmark to evaluate the numerical reasoning capabilities of large language models (LLMs) when analyzing specialized documents containing both text and tables. Your primary task is to propose **ONE** new, significantly more difficult question based on the provided document context and a set of existing, simpler questions. The new question must be fundamentally different and more complex than the provided examples, requiring both **document comprehension** and **multi-step mathematical reasoning** to arrive at a **single, non-zero numerical answer**. You must also provide the correct numerical answer and a **detailed step-by-step derivation** showing how the answer is obtained from the document.

## Principles for Question Design

Adhere strictly to the following principles when crafting your question, answer, and derivation:

1.  **Language Requirement**: Questions, answers, and derivations must be in English.

2.  **Complexity and Reasoning Depth**:
    - The question must be challenging, requiring the LLM to go beyond simple retrieval. It should not be solvable trivially or in a few inference steps.
    - It must involve **multi-step mathematical reasoning** (e.g., requiring two or more distinct calculation steps).
    - It should necessitate **integration of information** from different parts of the document (e.g., combining data from a table with information from a text paragraph, or using multiple rows/columns from a table).
    - Aspects like summarization or complex information extraction can be part of the process.

3. **Avoided Question Types**:
    - **Simple Counting**: Avoid questions like "How many X are there?" if X is easily countable or directly stated. If counting is involved as an intermediate step for a larger calculation and the count is small (<=10), it's acceptable.
    - **Direct Retrieval**: Avoid questions answerable by looking up a single, isolated piece of information.
    - **Excessive External Knowledge**: Questions should primarily rely on the provided document. Only common sense or minimal domain-specific knowledge (e.g., basic financial concepts like 'profit = revenue - cost' if contextually appropriate and derivable) inferable from the document is allowed.
    - **Artificially Convoluted Questions**: Questions should be natural and reflect realistic user inquiries. Do not create unnaturally complex or trick questions solely to increase difficulty.
    - **Multiple Numeric Answers Prohibited**: Questions must have a **single numeric value** as the final answer. Questions requiring two or more separate numeric outputs (e.g., “What are the revenues of A and B?”) are strictly forbidden. Intermediate steps may involve multiple numbers, but the final output must be one number.

4. **Escalate Question Difficulty**: The new question must demonstrate a higher order of reasoning than the Previous Examples. First, analyze the examples to identify their simple reasoning patterns (e.g., direct lookups, single calculations). Then, create a new question that incorporates one or more of the following advanced reasoning types:
    -  **Period-over-Period Calculation**: Requires calculating growth, decline, or change between different time periods.
    -  **Ratio or Metric Derivation**: Requires calculating a financial metric or ratio not explicitly stated in the document.
    -  **Aggregation and Filtering**: Requires aggregating data across multiple rows/columns/sections after filtering based on a text-based condition.
    -  **Projection or Implication**: Requires using data from the document to answer a "what if" or forward-looking question based only on the provided numbers.

5.  **Information Obscurity**:
    - Start with a clear question word (What/How/Where/When).
    - Do not explicitly mention or paraphrase key numerical values from the document within the question itself. The LLM should identify and extract these values.
    - Phrase questions to require inference and understanding of relationships between data points rather than just locating them.

6.  **Factual Grounding**:
    - All information required to answer the question must be present in or directly derivable from the provided document.
    - Do not introduce hypothetical scenarios, fictional data, or assumptions not supported by the document.
    - Questions should not contain any references to "Article 1", "Article 2", etc. They should be understandable without any additional context.

7.  **Numerical Answer**:
    - The final answer **must be a single non-zero numerical value**.
    - An answer with more than two numerical values is unacceptable.
    - If the document implies units (e.g., millions of dollars, percentages), the question should be phrased such that the numerical answer alone is sufficient (e.g., "What is the value in millions of dollars?" rather than expecting the answer to include "million dollars").

8.  **Step-by-Step Derivation**:
    - Provide a clear, step-by-step derivation for your answer.
    - This derivation must explicitly reference specific data points or phrases from the document (e.g., "From Table 3, Row 'Revenue', Year 2023..." or "As stated in paragraph 2...").
    - Detail all mathematical operations performed in each step. This helps verify the question's solvability and reasoning path.

## Output Format
Your response must conclude with a JSON object containing the following keys: "question" and "answer", placed after your reasoning. 

```json
{
  "question": "<A well-structured English question that adheres to all design principles>",
  "answer": "<A single, non-zero numerical answer>"
}
```

## Document Context

<text>
{context}
</text>

## Previous Examples
{examples}
"""

doc_mc_questioner_template_with_cache = """You are an expert in document analysis. We are building a benchmark to evaluate the capabilities of large language models (LLMs) on fact retrieval, reasoning across multiple constraints, and accurate synthesis of information into coherent responses. You will be provided with a document context and a set of simpler, existing questions. Your primary task is to generate **ONE** new, highly challenging multiple-choice question with one correct answer and three plausible, distinct distractors. The new question must be fundamentally different and more complex than the provided examples, requiring both **document comprehension** and **multi-hop reasoning**. Crucially, you must also provide a **detailed explanation** for why the correct answer is correct (including derivation steps) and why each distractor is incorrect.

## Principles for Question and Option Design

Adhere strictly to the following principles when crafting your question, answer, options, and derivation:

1.  **General Requirements**:
    - All questions, options, and explanations must be in **English**.
    - Questions should be **challenging**, requiring more than simple retrieval or a few inference steps.

2.  **Cognitive Complexity Requirements for the Question**:
    - Must necessitate **multi-step reasoning** (e.g., involving three or more distinct logical or calculation steps).
    - Should require the integration of **at least three distinct data points** from different parts of the document (e.g., combining data from a table with text, or using multiple rows/columns/cells).
    - Should demand the **synthesis of quantitative data with qualitative information** found in the text.
    - The problem setup should have the potential for common misinterpretations, which will inform distractor design.

3.  **Content Validity Criteria**:
    - The question and all options must be **exclusively answerable using information from the provided document**. No external knowledge beyond common sense or very basic, universally understood concepts (e.g., profit = revenue - cost, if directly applicable and data is provided) should be required.
    - If applicable to the document type (e.g., financial reports), prioritize questions with **regulatory/compliance implications** or those highlighting significant financial outcomes.
    - Ensure numerical values involved in the question or options require **contextual interpretation** within the document, not just direct look-up.
    - Avoid trivia; focus on questions that address **material information or key insights** derivable from the document.

4.  **Distractor Development Guidelines (for the 3 incorrect options)**:
    - Each of the **three distractors** must be plausible yet clearly incorrect upon careful analysis.
    - Distractors should represent **distinct error paths** or common misinterpretations.
    - At least one distractor should represent a **common conceptual misunderstanding** related to the document's content or how information is presented.

5.  **Forbidden Question/Option Patterns**:
    - **Simple Counting**: Avoid questions like "How many X are there?" if X is easily countable or directly stated. Small counts (<=5) as part of a larger calculation are acceptable.
    - **Direct Retrieval**: Avoid questions where the answer (or its direct components) can be found in a single, obvious location without further processing.
    - **Excessive External/Professional Knowledge**: Questions must not require significant domain-specific knowledge not provided or clearly inferable from the document.
    - **Artificially Convoluted Questions**: Questions should be natural and reflect realistic analytical tasks. Do not create unnaturally complex or trick questions solely for difficulty.
    - **No Fabricated Information**: Strictly adhere to document content. Do not introduce hypothetical scenarios, data, or assumptions not explicitly stated or directly inferable.
    - **Ambiguous Scenarios**: The question must have one unambiguously correct answer based *solely* on the provided document.
    - **Ethical Dilemmas or Subjective Interpretations**: Avoid questions requiring ethical judgment or subjective interpretation unless the document itself provides explicit criteria for such judgment.
    - **Vague Options**: All options, including distractors, must be precise and unambiguous.

6. **Escalate Question Difficulty**: The new question must demonstrate a higher order of reasoning than the Previous Examples. First, analyze the examples to identify their simple reasoning patterns (e.g., fact retrieval, single-step comparison). Then, create a new question that incorporates one or more of the following advanced reasoning types:
    - Quantitative Reasoning & Calculation: Requires performing mathematical operations (e.g., addition, subtraction, percentage change, averaging) on data from multiple sources.
    - Comparative & Superlative Analysis: Requires comparing multiple entities based on synthesized criteria to find the one that is highest, lowest, best, etc.
    - Conditional or Causal Reasoning: Structured as an "if-then" scenario or asks for the cause/effect of a situation by linking different documents (e.g., "What would be the total cost if the discount from Document A were applied to the price listed in Document B?").
    - Synthesis Across Data Types: Forces connection between qualitative information (e.g., a policy description) and quantitative data (e.g., a number in a table) to reach a conclusion.

7.  **Answer and Explanation Requirements**:
    - The correct answer must be <correct_answer>.
    - A **detailed derivation** for the correct answer must be provided, showing step-by-step calculations and referencing specific parts of the document (e.g., "From Table X, Row Y...", "As stated in paragraph Z...").
    - For **each distractor**, provide a brief explanation of why it is incorrect, ideally linking it to the type of error it represents (e.g., "Option A is incorrect because it omits the X deduction mentioned in...", "Option B results from incorrectly summing X and Y instead of finding their difference...").

## Output Format
Your response must conclude with a JSON object containing the following keys:  "question", "options", and "answer", placed after your reasoning. 

```json
{
  "question": "<A well-structured multiple choice English question, exclude choices and answer>",
  "options": {
    "A": "<Text for choice A>",
    "B": "<Text for choice B>",
    "C": "<Text for choice C>",
    "D": "<Text for choice D>"
  },
  "answer": "<correct_answer>"
}
```

## Document Context

<text>
{context}
</text>

## Previous Examples
{examples}
"""


QUESTIONER_PROMPT_MAPPING = {
    "docmath_qa": docmath_qa_questioner_template,
    "doc_mc": doc_mc_questioner_template,
    "doc_general_qa": doc_general_qa_questioner_template,
    "default": docmath_qa_questioner_template,
}

QUESTIONER_PROMPT_WITH_HISTORY_MAPPING = {
    "docmath_qa": docmath_qa_questioner_template_with_cache,
    "doc_mc": doc_mc_questioner_template_with_cache,
    "doc_general_qa": doc_general_qa_questioner_template_with_cache,
    "default": docmath_qa_questioner_template_with_cache,
}

RESPONDER_PROMPT_MAPPING = {
    "docmath_qa": docmath_qa_responder_template,
    "doc_mc": doc_mc_responder_template,
    "doc_general_qa": doc_general_qa_responder_template,
    "default": docmath_qa_responder_template,
}


