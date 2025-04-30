import os
import asyncio
import traceback
from events import CleaningInputRequiredEvent, CleaningResponseEvent
from workflow import DataAnalysisFlow

async def run_workflow(dataset_path):
    """Run the data analysis workflow on the given dataset"""

    workflow = DataAnalysisFlow(timeout=300, verbose=True)

    try:
        handler = workflow.run(
            dataset_path=dataset_path,
        )

       
        async for event in handler.stream_events():
            print(f"Run Workflow Loop: Received event: {type(event).__name__}")

            if isinstance(event, CleaningInputRequiredEvent):
                print("Run Workflow Loop: Handling CleaningInputRequiredEvent.")
                user_input_numbers = input(event.prompt_message) 

                print(f"Run Workflow Loop: User entered numbers: {user_input_numbers}")
                print("Run Workflow Loop: Sending CleaningResponseEvent...")
               
                handler.ctx.send_event(
                    CleaningResponseEvent(user_choices={"numbers": user_input_numbers.strip()})
                )
                print("Run Workflow Loop: Sent CleaningResponseEvent.")

        final_result_dict = await handler

        print("\n==== Final Report ====")
        final_report = final_result_dict.get('final_report', 'N/A')
        print(final_report)

        # Add visualization info
        viz_info = final_result_dict.get('visualization_info', 'No visualization info generated.')
        print("\n==== Visualization Info ====")
        print(viz_info)

        # Print plot paths if available
        plot_paths = final_result_dict.get('plot_paths', [])
        if plot_paths and isinstance(plot_paths, list):
            print("\nGenerated Plots:")
            for path in plot_paths:
                # Check if it's an error message from the tool
                if "Error:" not in path:
                    print(f"- {os.path.abspath(path)}") # Show absolute path
                else:
                    print(f"- {path}") # Print error message
        elif isinstance(plot_paths, str): # Handle case where string message is returned
             print(f"\nPlot Generation Note: {plot_paths}")

        return final_result_dict
    except Exception as e:
         print(f"Workflow failed: {e}")
         traceback.print_exc()
         return None

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        dataset_path = sys.argv[1]
    else:
        dataset_path = input("Enter the path to the dataset CSV file: ")
    
    # Ensure path uses correct path separators for the OS
    dataset_path = os.path.normpath(dataset_path)
    
    # Run the workflow
    asyncio.run(run_workflow(dataset_path))