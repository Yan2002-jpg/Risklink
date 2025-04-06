from __future__ import annotations
import asyncio
import uuid
import json
from pydantic import BaseModel
from agents import (
    Agent,
    Runner,
    handoff,
    trace,
    TResponseInputItem,
    MessageOutputItem,
    HandoffOutputItem,
    ItemHelpers,
)
from agents.extensions.handoff_prompt import RECOMMENDED_PROMPT_PREFIX
from questionnaire import questionnaire  # Expected to be a dict: {domain: [questions, ...]}

# -----------------------
# Shared Context Model
# -----------------------

class RiskAssessmentContext(BaseModel):
    current_domain: str | None = None
    current_question: str | None = None
    user_response: str | None = None

# -----------------------
# Report Agent
# -----------------------

class ReportAgent:
    def __init__(self, filename="risk_assessment_report.json"):
        self.filename = filename
        self.responses = []
    
    def validate_answer(self, response):
        """Checks if the response starts with YES, NO, or NOT APPLICABLE."""
        valid_starts = {"YES", "NO", "NOT APPLICABLE"}
    
        # Normalize the response (strip whitespace, lowercase, then uppercase first word)
        words = response.strip().split()  # Get first word safely
        first_word = words[0].upper().rstrip(",.") if words else "INVALID"  # Normalize punctuation
    
        return first_word if first_word in valid_starts else "INVALID"

    def record_answer(self, question, response):
        """Records an answer after validation, ensuring full answer is stored."""
        # Loop until the user provides a valid response.
        while True:
            validated_answer = self.validate_answer(response)
            if validated_answer == "INVALID":
                #print(f"Warning: Invalid response format for question: {question}")
                response = input(f"Please answer with 'YES', 'NO', or 'NOT APPLICABLE': ").strip()
            else:
                break
        
        # Only record the response once after it's validated
        structured_response = {
            "Question": question,
            "Answer": validated_answer,
            "Entire Answer": response.strip()
        }
        
        # Avoid duplicate entries: check if the question is already in the responses.
        if not any(item['Question'] == question for item in self.responses):
            self.responses.append(structured_response)
            print(f"Answer recorded: {response.strip()}")
        else:
            print(f"Answer for the question '{question}' has already been recorded.")
    
    def save_report(self):
        """Saves the responses to a JSON file."""
        with open(self.filename, "w") as f:
            json.dump(self.responses, f, indent=4)
        print(f"Report saved to {self.filename}")

# -----------------------
# Agent Definitions
# -----------------------

main_controller = Agent(
    name="MainController",
    instructions=f"""{RECOMMENDED_PROMPT_PREFIX}
You are the Main Orchestrator for a voice cyber risk assessment chatbot. Your role is to orchestrate the conversation the between the user and other AI agents.
You have access to three AI agents, the Question Agent, the Clarification Agent, and the Answer Checker Agent.

An example workflow of your first duty is as follows:
When the user is ready and a new question is to be asked, transfer control to the Question Agent who will ask the user a question from the cyber risk assessment.
When a user response is received, classify what kind of response it is given the question asked. 
Responses can either be classified as an answer to a question, or a clarification/followup question to the risk assessment question.
If the user response is classified as a clarification, you will transfer control to the Clarification Agent who will provide you a clarification to the user response.
After the Clarification Agent provides an explanation, re-ask the exact original risk assessment question.
If the user response is classified as an answer you will transfer control to the Answer Checker Agent, who will declare whether the user reponse is valid for the given question.
Only if the Answer Checker agent declares the user response is valid will you be able to transfer control to the Question Agent in order to ask the next question.
If the Answer Checker declares the user response to be invalid you will not be able to move on to the next question, even if the user asks you too.
Ultimately the answer checker will notify you when the user response is valid and you can then move on to the next question.
When classifying a users response self-reflect on how you classified it and your choice of the appropriate agent you will transfer control to.

Your secondary duty is to filter out any user response that is not relevant to the cyber risk assessment.
Irrelevant responses may include miscellaneous questions such as: What is the whether?
Other irrelevant responses may be statements such as: I like cars.
When filtering out user responses self-reflect on if the response is truly related to the risk assessment question
""",
    handoffs=[],  # Set below.
    output_type = RiskAssessmentContext
)

# Question Agent: Outputs the current question verbatim.
question_agent = Agent(
    name="QuestionAgent",
    instructions=f"""{RECOMMENDED_PROMPT_PREFIX}
You are the Question Agent within a cyber risk assessment. Your responsibility is to read in cyber risk assessment questions stored in the context.
These questions are split into various domains across cyber security.
You are to rephrase a question in a manner a novice to cybersecurity would understand.
Self reflect on your rephrased question to ensure that you retain all the points the orginal question was asking.
Do not ask any additional questions that were not stored in the context. Simply rephrase the exact text of the current question.
After outputting the question, immediately transfer control back to the Main Controller.""",
    handoffs=[handoff(agent=main_controller)],
    output_type = RiskAssessmentContext
)

# Clarification Agent: Provides a concise explanation of the current question.
clarification_agent = Agent(
    name="ClarificationAgent",
    instructions=f"""{RECOMMENDED_PROMPT_PREFIX}
You are an AI that is a domain level expert in all areas of cyber security and you are currently acting as a Clarification Agent within a cyber risk assessment. 
You will be provided a cyber risk assessment question and a user response.
Your role is to provide a brief and clear explanation to the user response in relation to the risk assessment question.
You are to state your explanation in a manner that a novice to cybersecurity would understand.
Your explanation should be no more than 10 sentences long.
Self reflect on your explanation on whether it specifically answers the question held in the users response and if the explanation is phrased in simple to understand terms.
DO NOT RETURN or modify the current question – simply explain it in simple language
After providing your explanation, immediately transfer control back to the Main Controller.""",
    handoffs=[handoff(agent=main_controller)],
    output_type = RiskAssessmentContext
)

answer_checker_agent = Agent[RiskAssessmentContext](
    name="AnswerCheckerAgent",
    instructions=f"""{RECOMMENDED_PROMPT_PREFIX}
You are the Answer Checker Agent. Your job is to evaluate whether the user's answer is relevant to the current question.
Respond with one of the following categories:
- "valid": if the user provided a reasonable, relevant answer.
- "clarification": if the user asked a question, showed confusion, or seems unsure.
- "off_topic": if the response appears like a joke, random text, or completely unrelated content.

You must only return one of these values as plain text — no explanation or formatting.
""",
    handoffs=[handoff(agent=main_controller)]
)

# Configure the Main Controller to hand off to the specialized agents.
main_controller.handoffs = [
    handoff(agent=question_agent),
    handoff(agent=clarification_agent),
    handoff(agent=answer_checker_agent),
]

# -----------------------
# Helper: Classify Answer
# -----------------------

def classify_answer(answer: str) -> str:
    answer = answer.strip().lower()
    question_words = {"what", "why", "how", "when", "where", "which", "who"}
    if "?" in answer:
        return "clarification"
    if answer.split() and answer.split()[0] in question_words:
        return "clarification"
    return "valid"

# -----------------------
# Main Conversation Loop
# -----------------------

async def run_question(domain: str, question: str, report_agent: ReportAgent):
    # Create a fresh context for this question.
    context = RiskAssessmentContext(
        current_domain=domain,
        current_question=question,
        user_response=None,
    )
    conversation_id = uuid.uuid4().hex[:16]
    current_agent = main_controller

    # Step 1: Ask the question using the Question Agent.
    input_items = [{"content": context.current_question, "role": "system"}]
    with trace("Risk Assessment", group_id=conversation_id):
        result = await Runner.run(main_controller, input_items, context=context)
    
    # Extract the output; if it doesn't match exactly the current question, override it.
    asked_question = ""
    for item in result.new_items:
        if isinstance(item, MessageOutputItem):
            asked_question = ItemHelpers.text_message_output(item).strip()
            break
    if asked_question != context.current_question.strip():
        # Override any extra output with the original question.
        asked_question = context.current_question.strip()
    print("Question:", asked_question)
    
    for item in result.new_items:
        if isinstance(item, HandoffOutputItem):
            print(f"Handoff: {item.source_agent.name} -> {item.target_agent.name}")
    
    current_agent = result.last_agent

    # Step 2: Wait for the user's answer.
    user_ans = input("Your answer: ").strip()
    context.user_response = user_ans
    classification = classify_answer(user_ans)

    # Loop until a valid answer is received (YES, NO, NOT APPLICABLE).
    while classification == "invalid":
        print("Invalid response. Please answer with 'YES', 'NO', or 'NOT APPLICABLE'.")
        user_ans = input("Your answer: ").strip()
        context.user_response = user_ans
        classification = classify_answer(user_ans)
    
    while classification == "clarification":
        # Step 3: Trigger the Clarification Agent for more details.
        clar_prompt = f"Clarify the question: {context.current_question}\nUser asked: {user_ans}"
        input_items = [{"content": clar_prompt, "role": "user"}]
        with trace("Risk Assessment", group_id=conversation_id):
            result = await Runner.run(main_controller, input_items, context=context)
        
        for item in result.new_items:
            if isinstance(item, MessageOutputItem):
                print("Clarification:", ItemHelpers.text_message_output(item))
            elif isinstance(item, HandoffOutputItem):
                print(f"Handoff: {item.source_agent.name} -> {item.target_agent.name}")

        # Step 4: Re-ask the original question exactly.
        input_items = [{"content": context.current_question, "role": "system"}]
        with trace("Risk Assessment", group_id=conversation_id):
            result = await Runner.run(main_controller, input_items, context=context)
        
        for item in result.new_items:
            if isinstance(item, MessageOutputItem):
                asked_question = ItemHelpers.text_message_output(item).strip()
                if asked_question != context.current_question.strip():
                    asked_question = context.current_question.strip()
                print("Question re-asked:", asked_question)
            elif isinstance(item, HandoffOutputItem):
                print(f"Handoff: {item.source_agent.name} -> {item.target_agent.name}")

        # Step 5: Get a new answer.
        user_ans = input("Your answer after clarification: ").strip()
        context.user_response = user_ans
        classification = classify_answer(user_ans)
    # print("THIS IS A TEST")
    # print(user_ans)
    # print(context.current_question)

    check_input = [{
    "content": f"Question: {context.current_question}\nAnswer: {user_ans}",
    "role": "user"
    }]
    with trace("Answer Checking", group_id=conversation_id):
        answer_checker_result = await Runner.run(answer_checker_agent, check_input, context=context)


    ACclassification = None
    for item in answer_checker_result.new_items:
        if isinstance(item, MessageOutputItem):
            ACclassification = ItemHelpers.text_message_output(item).strip().lower()
            print(ACclassification)
            #print("this is classification")
    #         print (classification)
    # while classification!="valid":
    #     user_ans = input("Your answer: ").strip()
    #     context.user_response = user_ans

    #     check_input = [{
    #         "content": f"Question: {context.current_question}\nAnswer: {user_ans}",
    #         "role": "user"
    #     }]
    #     with trace("Answer Checking", group_id=conversation_id):
    #         result = await Runner.run(answer_checker_agent, check_input, context=context)

    #     classification = None
    #     for item in result.new_items:
    #         if isinstance(item, MessageOutputItem):
    #             classification = ItemHelpers.text_message_output(item).strip().lower()

    #     if classification == "valid":
    #         print("Answer accepted:", user_ans)
    #         break

    #     elif classification == "off_topic":
    #         print("Your answer seems off-topic. Please answer based on your organization's practices.")
        
    #     elif classification == "clarification":
    #         print("It looks like you're asking a question or seem unsure. Let me help clarify it.")
    #         clar_prompt = f"Clarify the question: {context.current_question}\nUser asked: {user_ans}"
    #         input_items = [{"content": clar_prompt, "role": "user"}]
    #         with trace("Clarification", group_id=conversation_id):
    #             result = await Runner.run(clarification_agent, input_items, context=context)
    #         for item in result.new_items:
    #             if isinstance(item, MessageOutputItem):
    #                 print("Clarification:", ItemHelpers.text_message_output(item))

    #     print("Re-asking the question:")
    #     input_items = [{"content": context.current_question, "role": "system"}]
    #     with trace("Re-ask", group_id=conversation_id):
    #         result = await Runner.run(question_agent, input_items, context=context)
    #     for item in result.new_items:
    #         if isinstance(item, MessageOutputItem):
    #             print("Question:", ItemHelpers.text_message_output(item))

    # Step 6: Accept the valid answer and store it.
    # print("Answer accepted:", user_ans)
    # print("Final answer for question:", context.current_question, "->", context.user_response)

    # Step 7: Call the Report Agent to save the answer.
    if report_agent.record_answer(context.current_question, user_ans):
        print(f"Answer for question '{context.current_question}' recorded successfully.")

    # Step 8: Save the report after recording all answers.
    report_agent.save_report()


async def main():
    report_agent = ReportAgent()
    questions_list = [(domain, q) for domain, qs in questionnaire.items() for q in qs]
    for domain, q in questions_list:
        await run_question(domain, q, report_agent)
        print()
    report_agent.save_report()

if __name__ == "__main__":
    asyncio.run(main())