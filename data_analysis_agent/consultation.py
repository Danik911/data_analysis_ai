import os
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from llama_index.core.agent import FunctionCallingAgent
from llama_index.core.workflow import Context
from events import CleaningInputRequiredEvent, CleaningResponseEvent

async def generate_consultation_options(
    llm, 
    agent_suggestion: str,
    stats_summary: str,
    column_info: Dict
) -> str:
    """
    Generate numbered options for data cleaning based on analysis.
    
    Args:
        llm: The language model to use for the agent
        agent_suggestion: Suggestions from the data preparation agent
        stats_summary: Summary of data statistics
        column_info: Information about columns in the DataFrame
        
    Returns:
        Formatted string with numbered options for cleaning
    """
    print("[CONSULTATION] Generating consultation options")
    # Create consultation agent
    consultation_agent = FunctionCallingAgent.from_tools(
        tools=[], 
        llm=llm,
        verbose=True,
        system_prompt=(
            "You are a data cleaning assistant. You are given an initial analysis and suggested cleaning steps. "
            "Your task is to formulate concise, **numbered options** for the user based *only* on the issues explicitly identified in the analysis (missing values, outliers, duplicates, data quality). "
            "**If no issues were identified for a category (e.g., no missing values found), do NOT ask about it.** "
            "For each identified issue, present the finding and suggest 1-3 common handling strategies as numbered options (e.g., 1. Fill median, 2. Fill mean, 3. Drop rows). "
            "Start numbering options from 1 and continue sequentially across all issues. "
            "Combine these into a single, clear message asking the user to reply with the **numbers** of their chosen options, separated by semicolons. Use the provided analysis as context.\n"
            "Example Output Format (if missing values and outliers were found, but no duplicates or quality issues):\n"
            "Based on the analysis:\n"
            "Missing Values ('Time'): 3 found.\n"
            "  1. Fill median\n"
            "  2. Fill mean\n"
            "  3. Drop rows\n"
            "Outliers ('Distance'): Max 99.0 is high.\n"
            "  4. Keep outliers\n"
            "  5. Remove outlier rows\n"
            "  6. Cap outliers at 95th percentile\n"
            "Please reply with the numbers of your chosen options, separated by semicolons (e.g., '1;5'): "
        )
    )

    # Create prompt for consultation agent
    consultation_prompt = (
        f"Formulate numbered user questions based on this analysis/suggestion:\n"
        f"<analysis>\n{agent_suggestion}\n</analysis>\n\n"
        f"Additional Context:\n"
        f"Stats Summary:\n{stats_summary}\n"
        f"Column Info:\n{column_info}"
    )
    
    print(f"--- Prompting Consultation Agent ---\n{consultation_prompt[:500]}...\n---------------------------------")
    
    # Get response from consultation agent
    print("[CONSULTATION] Waiting for agent response...")
    agent_response = await consultation_agent.achat(consultation_prompt)
    print("[CONSULTATION] Received agent response")
    
    consultation_message = agent_response.response if hasattr(agent_response, 'response') else "Could not generate consultation message."
    
    print(f"--- Consultation Message ---\n{consultation_message}\n----------------------------")
    
    return consultation_message

async def handle_user_consultation(
    ctx: Context, 
    llm,
    agent_suggestion: str,
    stats_summary: str,
    column_info: Dict,
    original_path: str
) -> Dict[str, Any]:
    """
    Handle the human consultation process, including generating options, getting user input, 
    and translating user choices into a descriptive action plan.
    
    Args:
        ctx: The workflow context
        llm: The language model to use for the agent
        agent_suggestion: Suggestions from the data preparation agent
        stats_summary: Summary of data statistics
        column_info: Information about columns in the DataFrame
        original_path: Path to the original data file
        
    Returns:
        Dictionary containing user's choices and the translated action plan
    """
    print("[CONSULTATION] Starting user consultation process")
    
    # Generate consultation message with numbered options
    consultation_message = await generate_consultation_options(
        llm, 
        agent_suggestion,
        stats_summary,
        column_info
    )
    
    # Emit event to request user input
    issues_placeholder = {"message": consultation_message}  # Keep original message for context
    print("[CONSULTATION] Emitting CleaningInputRequiredEvent...")
    ctx.write_event_to_stream(
        CleaningInputRequiredEvent(
            issues=issues_placeholder,
            prompt_message=consultation_message 
        )
    )

    # Wait for user response (expecting numbers)
    print("[CONSULTATION] Waiting for CleaningResponseEvent...")
    response_event = await ctx.wait_for_event(CleaningResponseEvent)
    print("[CONSULTATION] Received CleaningResponseEvent.")
    
    user_input_numbers = response_event.user_choices.get("numbers", "")  # Get raw numeric string
    print(f"[CONSULTATION] User chose numbers: {user_input_numbers}")
    
    # Translate user selections into a descriptive plan
    print("[CONSULTATION] Creating translation agent")
    translation_agent = FunctionCallingAgent.from_tools(
        tools=[],
        llm=llm,
        verbose=True,
        system_prompt=(
            "You are given a text containing numbered options for data cleaning and a string containing the numbers selected by the user (separated by semicolons). "
            "Your task is to generate a clear, descriptive summary of the actions corresponding to the selected numbers. "
            "This summary will be used as instructions for another agent. "
            "Format the output as a list of actions.\n"
            "Example Input:\n"
            "Options Text: 'Based on the analysis:\\nMissing Values ('Time'): 3 found.\\n  1. Fill median\\n  2. Fill mean\\nOutliers ('Distance'): Max 99.0 is high.\\n  3. Keep outliers\\n  4. Remove outlier rows'\n"
            "Selected Numbers: '1;4'\n"
            "Example Output:\n"
            "Apply the following user-specified cleaning steps:\n"
            "- For missing values in 'Time', apply strategy: Fill median.\n"
            "- For outliers in 'Distance', apply strategy: Remove outlier rows.\n"
        )
    )

    translation_prompt = (
        f"Translate the selected numbers into a descriptive action plan.\n\n"
        f"Options Text:\n'''\n{consultation_message}\n'''\n\n"
        f"Selected Numbers: '{user_input_numbers}'\n\n"
        f"Generate the descriptive action plan:"
    )
    
    print(f"--- Prompting Translation Agent ---\n{translation_prompt[:500]}...\n---------------------------------")
    
    print("[CONSULTATION] Waiting for translation agent response...")
    translation_response = await translation_agent.achat(translation_prompt)
    print("[CONSULTATION] Received translation agent response")
    
    user_approved_description = translation_response.response if hasattr(translation_response, 'response') else f"Could not translate choices: {user_input_numbers}"

    # Handle potential empty description from translation agent
    if not user_approved_description.strip() or "Could not translate" in user_approved_description:
        print(f"[CONSULTATION WARNING] Translation agent failed or returned empty description. Using fallback.")
        user_approved_description = f"Apply user choices corresponding to numbers: {user_input_numbers} based on the options provided."

    print(f"--- Generated User-Approved Preparation Description ---\n{user_approved_description}\n---------------------------------------")
    
    print("[CONSULTATION] User consultation process completed")
    return {
        "user_choices": user_input_numbers,
        "user_approved_description": user_approved_description,
        "original_path": original_path
    }