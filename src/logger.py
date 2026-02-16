"""
Session logger for Waku - Records all user inputs and session data to log files.
"""

import json
import os
from datetime import datetime
from pathlib import Path


class SessionLogger:
    """Logs all user inputs and processed data during a Waku session."""
    
    def __init__(self, log_dir="logs"):
        """
        Initialise the session logger.
        
        Parameters:
        - log_dir (str): Directory where log files will be stored (default: "logs")
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Create a timestamped log file for this session
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_id = timestamp
        self.log_file = self.log_dir / f"session_{timestamp}.log"
        
        # Also create a JSON file for structured data
        self.json_log_file = self.log_dir / f"session_{timestamp}.json"
        
        # Initialise session data
        self.session_data = {
            "session_id": self.session_id,
            "start_time": datetime.now().isoformat(),
            "user_inputs": {},
            "parsed_data": {},
            "career_matches": [],
            "errors": [],
            "events": []
        }
        
        self._write_initial_log()
    
    def _write_initial_log(self):
        """Write initial session log entry."""
        with open(self.log_file, 'w', encoding='utf-8') as f:
            f.write(f"=== WAKU SESSION LOG ===\n")
            f.write(f"Session ID: {self.session_id}\n")
            f.write(f"Start Time: {datetime.now().isoformat()}\n")
            f.write(f"{'='*50}\n\n")
    
    def log_user_input(self, field_name, value):
        """
        Log a user input field.
        
        Parameters:
        - field_name (str): Name of the input field
        - value: The value entered by the user
        """
        timestamp = datetime.now().isoformat()
        
        # Store in session data
        self.session_data["user_inputs"][field_name] = {
            "value": self._serialise_value(value),
            "timestamp": timestamp
        }
        
        # Write to log file
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(f"[{timestamp}] USER INPUT: {field_name}\n")
            f.write(f"  Value: {self._format_value_for_log(value)}\n\n")
    
    def log_all_user_data(self, user_data):
        """
        Log all collected user data at once.
        
        Parameters:
        - user_data (dict): Complete user input dictionary
        """
        timestamp = datetime.now().isoformat()
        
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(f"[{timestamp}] === USER DATA SUMMARY ===\n")
            for field, value in user_data.items():
                f.write(f"  {field}: {self._format_value_for_log(value)}\n")
            f.write(f"\n")
        
        # Store in session data
        self.session_data["user_inputs"] = {
            k: {"value": self._serialise_value(v), "timestamp": timestamp}
            for k, v in user_data.items()
        }
    
    def log_pdf_upload(self, filename, text_length=None):
        """
        Log a PDF file upload.
        
        Parameters:
        - filename (str): Name of the uploaded PDF
        - text_length (int): Length of extracted text
        """
        timestamp = datetime.now().isoformat()
        
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(f"[{timestamp}] PDF UPLOADED: {filename}\n")
            if text_length:
                f.write(f"  Extracted text length: {text_length} characters\n")
            f.write(f"\n")
        
        self.session_data["events"].append({
            "event": "pdf_upload",
            "filename": filename,
            "text_length": text_length,
            "timestamp": timestamp
        })
    
    def log_parsed_data(self, data_type, parsed_data):
        """
        Log parsed data from resume or form.
        
        Parameters:
        - data_type (str): Type of data ("resume" or "form")
        - parsed_data (dict): The parsed data
        """
        timestamp = datetime.now().isoformat()
        
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(f"[{timestamp}] PARSED {data_type.upper()} DATA:\n")
            for key, value in parsed_data.items():
                if key != "raw_text":  # Don't log full raw text to keep file readable
                    f.write(f"  {key}: {self._format_value_for_log(value)}\n")
            f.write(f"\n")
        
        self.session_data["parsed_data"][data_type] = {
            "data": self._serialise_value(parsed_data),
            "timestamp": timestamp
        }
    
    def log_career_matches(self, ranked_careers, top_n=10):
        """
        Log the ranked career matches.
        
        Parameters:
        - ranked_careers (list): List of career dictionaries ranked by match score
        - top_n (int): Number of top matches to log (default: 10)
        """
        timestamp = datetime.now().isoformat()
        
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(f"[{timestamp}] === CAREER MATCHES ===\n")
            for i, career in enumerate(ranked_careers[:top_n], 1):
                f.write(f"  {i}. {career['title']}\n")
            f.write(f"\n")
        
        # Store top matches in session data
        self.session_data["career_matches"] = [
            {
                "rank": i,
                "title": career["title"],
                "timestamp": timestamp
            }
            for i, career in enumerate(ranked_careers[:top_n], 1)
        ]
    
    def log_extracted_skills(self, skills_type, skills_list):
        """
        Log extracted skills.
        
        Parameters:
        - skills_type (str): Type of skills ("hard_skills", "soft_skills", etc.)
        - skills_list (list): List of extracted skills
        """
        timestamp = datetime.now().isoformat()
        
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(f"[{timestamp}] EXTRACTED {skills_type.upper()}: {len(skills_list)} found\n")
            if skills_list:
                f.write(f"  {', '.join(skills_list)}\n")
            f.write(f"\n")
        
        self.session_data["events"].append({
            "event": f"extracted_{skills_type}",
            "count": len(skills_list),
            "items": skills_list,
            "timestamp": timestamp
        })
    
    def log_error(self, error_message, error_type="general"):
        """
        Log an error that occurred during the session.
        
        Parameters:
        - error_message (str): Description of the error
        - error_type (str): Type of error (default: "general")
        """
        timestamp = datetime.now().isoformat()
        
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(f"[{timestamp}] ERROR ({error_type}): {error_message}\n\n")
        
        self.session_data["errors"].append({
            "type": error_type,
            "message": error_message,
            "timestamp": timestamp
        })
    
    def log_summary_generated(self, summary_text=None, length=None):
        """
        Log when a summary is generated.
        
        Parameters:
        - summary_text (str): The generated summary text
        - length (int): Length of generated summary
        """
        timestamp = datetime.now().isoformat()
        
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(f"[{timestamp}] SUMMARY GENERATED\n")
            if length:
                f.write(f"  Length: {length} characters\n")
            if summary_text and len(summary_text) < 200:
                f.write(f"  Content: {summary_text}\n")
            f.write(f"\n")
        
        self.session_data["events"].append({
            "event": "summary_generated",
            "length": length,
            "timestamp": timestamp
        })
    
    def log_event(self, event_name, details=None):
        """
        Log a general event during the session.
        
        Parameters:
        - event_name (str): Name of the event
        - details (dict): Optional details about the event
        """
        timestamp = datetime.now().isoformat()
        
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(f"[{timestamp}] EVENT: {event_name}\n")
            if details:
                for key, value in details.items():
                    f.write(f"  {key}: {self._format_value_for_log(value)}\n")
            f.write(f"\n")
        
        event_data = {
            "event": event_name,
            "timestamp": timestamp
        }
        if details:
            event_data.update(details)
        
        self.session_data["events"].append(event_data)
    
    def finalise(self):
        """
        Finalise the session log and save JSON data.
        Should be called at the end of the session.
        """
        timestamp = datetime.now().isoformat()
        self.session_data["end_time"] = timestamp
        
        # Write final entry to log file
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(f"\n{'='*50}\n")
            f.write(f"Session End Time: {timestamp}\n")
            f.write(f"Log file: {self.log_file}\n")
        
        # Save structured JSON data
        with open(self.json_log_file, 'w', encoding='utf-8') as f:
            json.dump(self.session_data, f, indent=2, default=str)
    
    @staticmethod
    def _serialise_value(value):
        """Convert a value to a serialisable format."""
        if isinstance(value, (str, int, float, bool, type(None))):
            return value
        elif isinstance(value, (list, dict)):
            return value
        else:
            return str(value)
    
    @staticmethod
    def _format_value_for_log(value):
        """Format a value for readable log output."""
        if isinstance(value, list):
            return ", ".join(str(v) for v in value) if value else "[empty]"
        elif isinstance(value, dict):
            return str(value)
        elif isinstance(value, str):
            return value if value.strip() else "[empty]"
        else:
            return str(value)


# Global logger instance
_session_logger = None


def get_session_logger(log_dir="logs"):
    """
    Get or create the global session logger instance.
    
    Parameters:
    - log_dir (str): Directory for log files (default: "logs")
    
    Returns:
    - SessionLogger: The global session logger instance
    """
    global _session_logger
    if _session_logger is None:
        _session_logger = SessionLogger(log_dir)
    return _session_logger


def reset_session_logger():
    """Reset the global session logger (for testing purposes)."""
    global _session_logger
    _session_logger = None
