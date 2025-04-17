from llama_index.core.workflow import Event

class DataPrepEvent(Event):
    original_path: str
    column_names: list[str]
    stats_summary: str
    column_info: dict 
    
class DataAnalysisEvent(Event):
    prepared_data_description: str
    original_path: str

class ModificationRequestEvent(Event):
    """Carries the user-approved modification description."""
    user_approved_description: str
    original_path: str

class ModificationCompleteEvent(Event):
    original_path: str
    modification_summary: str | None = None 

class InitialAssessmentEvent(Event):
    """Carries initial stats summary after loading data."""
    stats_summary: str
    column_info: dict 
    original_path: str

class CleaningInputRequiredEvent(Event):
    """Event indicating user input is needed for cleaning decisions."""
    issues: dict 
    prompt_message: str

class CleaningResponseEvent(Event):
    """Event carrying the user's cleaning decisions."""
    user_choices: dict

class VisualizationRequestEvent(Event):
    """Event indicating that data visualization is requested."""
    modified_data_path: str 
    report: str             
