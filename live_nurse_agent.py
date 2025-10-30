#!/usr/bin/env python3
"""
Live Microphone Fever Proforma Nurse Agent
==========================================

This is the complete fever proforma nurse agent system with live microphone support.
It provides both traditional form-based input and natural language speech input.
"""

import openai
import json
import os
import time
import tempfile
import wave
from datetime import datetime
from typing import Dict, Any
from dotenv import load_dotenv

# Try to import speech recognition, handle gracefully if not available
try:
    import speech_recognition as sr
    SPEECH_AVAILABLE = True
except ImportError:
    SPEECH_AVAILABLE = False
    print("⚠️ Speech recognition not available. Install with: pip install speechrecognition pyaudio")

# Try to import audio processing libraries
try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False
    print("⚠️ PyAudio not available for Whisper recording")

# Load environment variables from .env file
load_dotenv()

class FeverProformaNurseAgent:
    def __init__(self, api_key: str = None):
        """
        Initialize the Nurse Agent with OpenAI API key
        
        Args:
            api_key (str, optional): OpenAI API key. If not provided, will try to load from .env file
        """
        if api_key is None:
            api_key = os.getenv('API_KEY')
            if not api_key:
                raise ValueError("API key not found. Please provide it as parameter or set API_KEY in .env file")
        
        self.client = openai.OpenAI(api_key=api_key)
        
        # Initialize speech recognition only if available
        if SPEECH_AVAILABLE:
            try:
                self.recognizer = sr.Recognizer()
                self.microphone = sr.Microphone()
                self.speech_enabled = True
                self.stop_listening = None
                self.is_recording_active = False
                self.recorded_audio_chunks = []
                self.full_transcription = ""
                print("🎤 Live microphone initialized successfully!")
            except Exception as e:
                print(f"⚠️ Speech recognition hardware not available: {e}")
                self.speech_enabled = False
        else:
            self.speech_enabled = False
        
        # Initialize Whisper recording variables
        self.whisper_recording = False
        self.whisper_frames = []
        self.whisper_audio_stream = None
        self.whisper_pyaudio = None
        
        # Audio settings for Whisper
        self.WHISPER_FORMAT = pyaudio.paInt16 if PYAUDIO_AVAILABLE else None
        self.WHISPER_CHANNELS = 1
        self.WHISPER_RATE = 16000  # 16kHz for Whisper
        self.WHISPER_CHUNK = 1024
        
    def collect_patient_data(self) -> Dict[str, Any]:
        """
        Collect patient information through manual text input
        
        Returns:
            Dict containing all patient responses
        """
        print("=== FEVER PROFORMA - PATIENT DATA COLLECTION ===\n")
        
        patient_data = {}
        
        # Patient Information
        print("PATIENT INFORMATION:")
        patient_data['name'] = input("Name: ")
        patient_data['age'] = input("Age: ")
        patient_data['gender'] = input("Gender: ")
        patient_data['date'] = input("Date (or press Enter for today): ") or datetime.now().strftime("%Y-%m-%d")
        patient_data['occupation'] = input("Occupation: ")
        patient_data['address'] = input("Address: ")
        
        # Chief Complaint
        print("\nCHIEF COMPLAINT:")
        patient_data['fever_present'] = input("Fever present (Yes/No): ")
        patient_data['duration'] = input("Duration (days/weeks): ")
        patient_data['onset'] = input("Onset (Sudden/Gradual): ")
        
        # History of Presenting Illness
        print("\nHISTORY OF PRESENTING ILLNESS:")
        print("1. Characteristics of Fever:")
        patient_data['fever_frequency'] = input("   Frequency (Constantly present/Comes and goes): ")
        patient_data['fever_timing'] = input("   Timing (Day/Night/Both): ")
        patient_data['max_temperature'] = input("   Maximum temperature (°C/°F): ")
        
        print("\n2. Associated Symptoms:")
        symptoms = [
            'chills_shivering', 'sweating', 'fatigue', 'headache', 'muscle_pain',
            'joint_pain', 'rash', 'cough', 'sore_throat', 'nasal_discharge',
            'abdominal_pain', 'nausea_vomiting', 'diarrhea', 'urinary_symptoms'
        ]
        
        for symptom in symptoms:
            readable_symptom = symptom.replace('_', ' ').title()
            patient_data[symptom] = input(f"   {readable_symptom} (Yes/No): ")
            
            # Special cases for additional details
            if symptom == 'rash' and patient_data[symptom].lower() == 'yes':
                patient_data['rash_description'] = input("     Describe rash: ")
            elif symptom == 'cough' and patient_data[symptom].lower() == 'yes':
                patient_data['sputum_present'] = input("     Sputum present (Yes/No): ")
        
        patient_data['fever_progression'] = input("\n3. Fever progression (improving/worsening/unchanged): ")
        
        # Past Medical History
        print("\nPAST MEDICAL HISTORY:")
        patient_data['chronic_illnesses'] = input("Chronic illnesses (diabetes, hypertension, TB, HIV, etc.): ")
        patient_data['previous_fever_infections'] = input("Previous fever/infection history: ")
        
        # Medication History
        print("\nMEDICATION HISTORY:")
        patient_data['allergies'] = input("Allergies (Yes/No and list): ")
        patient_data['current_medications'] = input("Current medications: ")
        patient_data['fever_treatment_history'] = input("Previous fever treatment (antibiotics, antipyretics, etc.): ")
        
        # Social History
        print("\nSOCIAL HISTORY:")
        patient_data['smoking'] = input("Smoking (Yes/No): ")
        patient_data['alcohol'] = input("Alcohol (Yes/No): ")
        patient_data['travel_history'] = input("Travel history in last 6 months (Yes/No and details): ")
        patient_data['living_conditions'] = input("Living conditions (overcrowding, sanitation, etc.): ")
        
        # Exposure History
        print("\nEXPOSURE HISTORY:")
        patient_data['contact_with_sick'] = input("Recent contact with sick individuals (Yes/No and details): ")
        patient_data['mosquito_exposure'] = input("Mosquito bites/vector exposure (Yes/No and details): ")
        patient_data['contaminated_food_water'] = input("Exposure to contaminated food/water (Yes/No and details): ")
        
        # Additional Notes
        print("\nADDITIONAL NOTES:")
        patient_data['additional_notes'] = input("Any additional notes: ")
        
        return patient_data
    
    def format_patient_data_for_llm(self, patient_data: Dict[str, Any]) -> str:
        """
        Format patient data into a structured text for LLM analysis
        
        Args:
            patient_data (Dict): Patient data dictionary
            
        Returns:
            str: Formatted patient data
        """
        formatted_data = f"""
FEVER PROFORMA - PATIENT DATA

PATIENT INFORMATION:
- Name: {patient_data.get('name', 'Not provided')}
- Age: {patient_data.get('age', 'Not provided')}
- Gender: {patient_data.get('gender', 'Not provided')}
- Date: {patient_data.get('date', 'Not provided')}
- Occupation: {patient_data.get('occupation', 'Not provided')}
- Address: {patient_data.get('address', 'Not provided')}

CHIEF COMPLAINT:
- Fever Present: {patient_data.get('fever_present', 'Not provided')}
- Duration: {patient_data.get('duration', 'Not provided')}
- Onset: {patient_data.get('onset', 'Not provided')}

HISTORY OF PRESENTING ILLNESS:
1. Characteristics of Fever:
   - Frequency: {patient_data.get('fever_frequency', 'Not provided')}
   - Timing: {patient_data.get('fever_timing', 'Not provided')}
   - Maximum Temperature: {patient_data.get('max_temperature', 'Not provided')}

2. Associated Symptoms:
   - Chills and Shivering: {patient_data.get('chills_shivering', 'Not provided')}
   - Sweating: {patient_data.get('sweating', 'Not provided')}
   - Fatigue: {patient_data.get('fatigue', 'Not provided')}
   - Headache: {patient_data.get('headache', 'Not provided')}
   - Muscle Pain: {patient_data.get('muscle_pain', 'Not provided')}
   - Joint Pain: {patient_data.get('joint_pain', 'Not provided')}
   - Rash: {patient_data.get('rash', 'Not provided')}
   {f"   - Rash Description: {patient_data.get('rash_description', '')}" if patient_data.get('rash_description') else ""}
   - Cough: {patient_data.get('cough', 'Not provided')}
   {f"   - Sputum Present: {patient_data.get('sputum_present', '')}" if patient_data.get('sputum_present') else ""}
   - Sore Throat: {patient_data.get('sore_throat', 'Not provided')}
   - Nasal Discharge: {patient_data.get('nasal_discharge', 'Not provided')}
   - Abdominal Pain: {patient_data.get('abdominal_pain', 'Not provided')}
   - Nausea/Vomiting: {patient_data.get('nausea_vomiting', 'Not provided')}
   - Diarrhea: {patient_data.get('diarrhea', 'Not provided')}
   - Urinary Symptoms: {patient_data.get('urinary_symptoms', 'Not provided')}

3. Fever Progression: {patient_data.get('fever_progression', 'Not provided')}

PAST MEDICAL HISTORY:
- Chronic Illnesses: {patient_data.get('chronic_illnesses', 'Not provided')}
- Previous Fever/Infections: {patient_data.get('previous_fever_infections', 'Not provided')}

MEDICATION HISTORY:
- Allergies: {patient_data.get('allergies', 'Not provided')}
- Current Medications: {patient_data.get('current_medications', 'Not provided')}
- Fever Treatment History: {patient_data.get('fever_treatment_history', 'Not provided')}

SOCIAL HISTORY:
- Smoking: {patient_data.get('smoking', 'Not provided')}
- Alcohol: {patient_data.get('alcohol', 'Not provided')}
- Travel History: {patient_data.get('travel_history', 'Not provided')}
- Living Conditions: {patient_data.get('living_conditions', 'Not provided')}

EXPOSURE HISTORY:
- Contact with Sick Individuals: {patient_data.get('contact_with_sick', 'Not provided')}
- Mosquito/Vector Exposure: {patient_data.get('mosquito_exposure', 'Not provided')}
- Contaminated Food/Water Exposure: {patient_data.get('contaminated_food_water', 'Not provided')}

ADDITIONAL NOTES:
{patient_data.get('additional_notes', 'None provided')}
"""
        return formatted_data.strip()
    
    def generate_diagnosis_and_next_steps(self, patient_data_text: str) -> str:
        """
        Use OpenAI GPT to analyze patient data and generate provisional diagnosis and next steps
        
        Args:
            patient_data_text (str): Formatted patient data
            
        Returns:
            str: Provisional diagnosis and next steps
        """
        
        system_prompt = """You are an experienced nurse practitioner working in an Indian Outpatient Department (OPD). You have extensive experience in diagnosing common causes of fever in the Indian healthcare context.

Your task is to analyze the fever proforma data and provide:
1. A provisional diagnosis (or differential diagnoses if multiple possibilities exist)
2. Clear next steps for patient management

Consider the following in your analysis:
- Common fever causes in India (viral infections, bacterial infections, malaria, dengue, typhoid, UTI, respiratory infections, etc.)
- Patient's age, occupation, and living conditions
- Symptom patterns and progression
- Exposure history and risk factors
- Travel history and vector exposure
- Social and environmental factors

Provide your response in a clear, professional format that would be appropriate for medical documentation. Be specific about recommended investigations, treatments, and follow-up care.

If the information is insufficient for a clear diagnosis, mention what additional information or tests would be helpful."""

        user_prompt = f"""Please analyze this fever proforma data and provide a provisional diagnosis and next steps:

{patient_data_text}

Please structure your response as:

PROVISIONAL DIAGNOSIS AND NEXT STEPS:

[Your analysis here]"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=1500,
                temperature=0.3
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Error generating diagnosis: {str(e)}"

    def start_continuous_recording(self):
        """
        Start continuous recording session
        
        Returns:
            tuple: (success, message)
        """
        if not self.speech_enabled:
            return False, "Speech recognition not available. Please install: pip install speechrecognition pyaudio"
        
        try:
            print("\n🎤 Starting continuous recording session...")
            print("🔴 Recording active - speak now!")
            print("⏹️  Use stop_recording() to end session")
            
            # Stop any existing recording first
            if hasattr(self, 'stop_listening') and self.stop_listening:
                try:
                    self.stop_listening(wait_for_stop=False)
                except:
                    pass
            
            # Initialize recording state
            self.recorded_audio_chunks = []
            self.is_recording_active = True
            self.full_transcription = ""
            
            # Configure recognizer for better continuous capture
            self.recognizer.energy_threshold = 200  # Lower threshold for quieter speech
            self.recognizer.dynamic_energy_threshold = True
            self.recognizer.pause_threshold = 0.5  # Shorter pause detection
            self.recognizer.operation_timeout = None
            self.recognizer.phrase_time_limit = None  # No phrase limit
            
            print("🔧 Adjusting for ambient noise...")
            
            # Use a fresh microphone instance to avoid context manager conflicts
            mic = sr.Microphone()
            with mic as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=0.3)
            
            print("✅ Ready! Recording started - speak continuously...")
            
            # Start background listening with shorter chunks for better capture
            self.stop_listening = self.recognizer.listen_in_background(
                mic, 
                self._audio_callback,
                phrase_time_limit=3  # Process in 3-second chunks for continuous capture
            )
            
            return True, "Recording started successfully"
            
        except Exception as e:
            return False, f"Error starting recording: {str(e)}"
    
    def _audio_callback(self, recognizer, audio):
        """
        Callback function for continuous audio processing - processes each chunk immediately
        """
        try:
            # Only process audio if we're actively recording
            if hasattr(self, 'is_recording_active') and self.is_recording_active:
                # Store the raw audio chunk
                if not hasattr(self, 'recorded_audio_chunks'):
                    self.recorded_audio_chunks = []
                self.recorded_audio_chunks.append(audio)
                
                # Immediately transcribe this chunk and add to full transcription
                try:
                    chunk_text = recognizer.recognize_google(audio, language='en-IN')
                    if chunk_text.strip():
                        if not hasattr(self, 'full_transcription'):
                            self.full_transcription = ""
                        
                        # Add this chunk to the full transcription with a space
                        if self.full_transcription:
                            self.full_transcription += " " + chunk_text
                        else:
                            self.full_transcription = chunk_text
                        
                        print(f"📝 Captured: {chunk_text}")
                        
                except sr.UnknownValueError:
                    # This chunk couldn't be understood, but keep recording
                    pass
                except sr.RequestError as e:
                    print(f"⚠️ Recognition error for chunk: {e}")
                    
        except Exception as e:
            print(f"Audio callback error: {e}")
    
    def stop_continuous_recording(self):
        """
        Stop continuous recording and return the complete transcribed text
        
        Returns:
            str: Complete transcribed text from the entire recording session
        """
        if not self.speech_enabled:
            return "Speech recognition not available"
        
        try:
            # Mark recording as inactive
            self.is_recording_active = False
            
            # Stop listening
            if hasattr(self, 'stop_listening') and self.stop_listening:
                self.stop_listening(wait_for_stop=True)
                print("🛑 Recording stopped")
                self.stop_listening = None
            
            # Give a moment for any final chunks to process
            time.sleep(0.5)
            
            # Return the complete transcription that was built during recording
            if hasattr(self, 'full_transcription') and self.full_transcription:
                complete_text = self.full_transcription.strip()
                print(f"✅ Complete transcription: {complete_text[:100]}{'...' if len(complete_text) > 100 else ''}")
                print(f"📊 Total length: {len(complete_text)} characters")
                
                # Clean up for next recording
                self.full_transcription = ""
                self.recorded_audio_chunks = []
                
                if not complete_text:
                    return "No speech detected in recording. Please try again and speak clearly."
                
                return complete_text
            else:
                # Fallback: try to process any remaining audio chunks
                if hasattr(self, 'recorded_audio_chunks') and self.recorded_audio_chunks:
                    print(f"🔄 Processing {len(self.recorded_audio_chunks)} remaining audio chunks...")
                    
                    # Combine all chunks for a final attempt
                    combined_text = ""
                    for i, audio_chunk in enumerate(self.recorded_audio_chunks):
                        try:
                            chunk_text = self.recognizer.recognize_google(audio_chunk, language='en-IN')
                            if chunk_text.strip():
                                if combined_text:
                                    combined_text += " " + chunk_text
                                else:
                                    combined_text = chunk_text
                        except:
                            continue
                    
                    # Clear chunks
                    self.recorded_audio_chunks = []
                    
                    if combined_text.strip():
                        return combined_text
                    else:
                        return "No speech could be transcribed. Please try again and speak clearly."
                else:
                    return "No audio was recorded. Please try again and speak during the recording."
                
        except Exception as e:
            return f"Error stopping recording: {str(e)}"
    
    # ============== WHISPER AI RECORDING METHODS ==============
    
    def start_whisper_recording(self):
        """
        Start recording audio for Whisper AI transcription
        
        Returns:
            tuple: (success: bool, message: str)
        """
        if not PYAUDIO_AVAILABLE:
            return False, "PyAudio not available for Whisper recording"
        
        if self.whisper_recording:
            return False, "Already recording with Whisper"
        
        try:
            # Initialize PyAudio
            self.whisper_pyaudio = pyaudio.PyAudio()
            
            # Open audio stream
            self.whisper_audio_stream = self.whisper_pyaudio.open(
                format=self.WHISPER_FORMAT,
                channels=self.WHISPER_CHANNELS,
                rate=self.WHISPER_RATE,
                input=True,
                frames_per_buffer=self.WHISPER_CHUNK
            )
            
            # Reset frames
            self.whisper_frames = []
            self.whisper_recording = True
            
            print("🎤 Whisper recording started...")
            return True, "Whisper recording started successfully"
            
        except Exception as e:
            return False, f"Error starting Whisper recording: {str(e)}"
    
    def record_audio_continuously(self):
        """
        Continuously record audio frames while recording is active
        
        Returns:
            bool: True if successful, False if should stop
        """
        if not self.whisper_recording or not self.whisper_audio_stream:
            return False
        
        try:
            data = self.whisper_audio_stream.read(self.WHISPER_CHUNK, exception_on_overflow=False)
            self.whisper_frames.append(data)
            return True
        except Exception as e:
            print(f"Error recording audio: {e}")
            return False
    
    def stop_whisper_recording(self):
        """
        Stop Whisper recording and transcribe using OpenAI Whisper API
        
        Returns:
            str: Transcribed text
        """
        if not self.whisper_recording:
            return "No Whisper recording in progress"
        
        try:
            # Stop recording
            self.whisper_recording = False
            
            # Close audio stream
            if self.whisper_audio_stream:
                self.whisper_audio_stream.stop_stream()
                self.whisper_audio_stream.close()
                self.whisper_audio_stream = None
            
            # Close PyAudio
            if self.whisper_pyaudio:
                self.whisper_pyaudio.terminate()
                self.whisper_pyaudio = None
            
            print("🛑 Whisper recording stopped")
            
            # Check if we have audio data
            if not self.whisper_frames:
                return "No audio data recorded"
            
            # Create temporary WAV file for Whisper API
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
                temp_filename = temp_audio.name
                
                # Write WAV file
                with wave.open(temp_filename, 'wb') as wf:
                    wf.setnchannels(self.WHISPER_CHANNELS)
                    wf.setsampwidth(2)  # 16-bit audio = 2 bytes
                    wf.setframerate(self.WHISPER_RATE)
                    wf.writeframes(b''.join(self.whisper_frames))
            
            print("🔄 Transcribing with OpenAI Whisper...")
            
            # Transcribe using OpenAI Whisper API
            try:
                with open(temp_filename, "rb") as audio_file:
                    transcript = self.client.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio_file,
                        language="en"
                    )
                
                transcribed_text = transcript.text.strip()
                print(f"✅ Whisper transcription completed: {len(transcribed_text)} characters")
                
                # Clean up temp file
                os.unlink(temp_filename)
                
                # Reset frames
                self.whisper_frames = []
                
                if not transcribed_text:
                    return "No speech detected in recording"
                
                return transcribed_text
                
            except Exception as e:
                # Clean up temp file on error
                if os.path.exists(temp_filename):
                    os.unlink(temp_filename)
                return f"Error transcribing with Whisper: {str(e)}"
                
        except Exception as e:
            return f"Error stopping Whisper recording: {str(e)}"
    
    def collect_patient_data_via_speech(self) -> Dict[str, Any]:
        """
        Collect patient information through speech input
        
        Returns:
            Dict containing patient data extracted from speech
        """
        print("=== FEVER PROFORMA - SPEECH DATA COLLECTION ===\n")
        
        if not self.speech_enabled:
            print("⚠️ Live microphone not available.")
            print("📝 Falling back to text input mode...")
            print("Please type the patient information as if you were speaking:")
            print("Example: 'Patient John Smith, 35 years old, has fever for 3 days...'")
            
            speech_text = input("\nPatient description: ")
        else:
            print("🎤 LIVE MICROPHONE ACTIVATED")
            print("🗣️ You will be asked to provide patient information through speech.")
            print("Please speak clearly and provide as much detail as possible.\n")
            
            # Get patient's natural language description
            input("Press Enter when ready to start recording...")
            
            print("\n📋 Please provide the following information about the patient:")
            print("- Patient's basic information (name, age, gender, occupation, address)")
            print("- Chief complaint and fever details")
            print("- All symptoms the patient is experiencing")
            print("- Medical history, medications, and allergies")
            print("- Social history and recent exposures")
            print("- Any additional relevant information")
            print("\nYou can speak naturally - just describe the patient's condition as you would to a colleague.")
            
            # Record the patient description using continuous recording
            print("\n🎤 Starting recording... Click when you're done speaking.")
            success, message = self.start_continuous_recording()
            
            if success:
                input("Recording started! Speak now, then press Enter when finished...")
                speech_text = self.stop_continuous_recording()
            else:
                speech_text = f"Error starting recording: {message}"
            
            if speech_text.startswith("Error") or speech_text.startswith("Could not"):
                print(f"❌ {speech_text}")
                # Fallback to text input
                print("\n📝 Falling back to text input. Please type the patient information:")
                speech_text = input("Patient information: ")
        
        print(f"\n✅ Captured information: {len(speech_text)} characters")
        
        # Process the speech text with GPT to extract structured data
        return self.extract_patient_data_from_speech(speech_text)
    
    def extract_patient_data_from_speech(self, speech_text: str) -> Dict[str, Any]:
        """
        Use GPT to extract structured patient data from natural language speech
        
        Args:
            speech_text (str): Transcribed speech about patient
            
        Returns:
            Dict containing structured patient data
        """
        
        extraction_prompt = f"""
You are a medical assistant helping to extract structured patient information from a natural language description. 

Please extract the following information from the given text and format it as JSON. If information is not provided, use "Not mentioned" as the value.

Required fields:
- name: Patient's name
- age: Patient's age
- gender: Patient's gender  
- date: Date of assessment (use today's date if not mentioned)
- occupation: Patient's occupation
- address: Patient's address
- fever_present: Does patient have fever (Yes/No)
- duration: How long has fever been present
- onset: Fever onset (Sudden/Gradual)
- fever_frequency: Fever frequency (Constantly present/Comes and goes)
- fever_timing: When fever occurs (Day/Night/Both)
- max_temperature: Maximum temperature recorded
- chills_shivering: Presence of chills/shivering (Yes/No)
- sweating: Presence of sweating (Yes/No)
- fatigue: Presence of fatigue (Yes/No)
- headache: Presence of headache (Yes/No)
- muscle_pain: Presence of muscle pain (Yes/No)
- joint_pain: Presence of joint pain (Yes/No)
- rash: Presence of rash (Yes/No)
- rash_description: Description of rash if present
- cough: Presence of cough (Yes/No)
- sputum_present: If cough present, is sputum present (Yes/No)
- sore_throat: Presence of sore throat (Yes/No)
- nasal_discharge: Presence of nasal discharge (Yes/No)
- abdominal_pain: Presence of abdominal pain (Yes/No)
- nausea_vomiting: Presence of nausea/vomiting (Yes/No)
- diarrhea: Presence of diarrhea (Yes/No)
- urinary_symptoms: Presence of urinary symptoms (Yes/No)
- fever_progression: How fever is progressing (improving/worsening/unchanged)
- chronic_illnesses: Any chronic medical conditions
- previous_fever_infections: Previous fever or infection history
- allergies: Known allergies
- current_medications: Current medications
- fever_treatment_history: Previous fever treatments tried
- smoking: Smoking history (Yes/No)
- alcohol: Alcohol consumption (Yes/No)
- travel_history: Recent travel history
- living_conditions: Living conditions
- contact_with_sick: Recent contact with sick individuals
- mosquito_exposure: Mosquito or vector exposure
- contaminated_food_water: Exposure to contaminated food/water
- additional_notes: Any additional relevant information

Patient description: "{speech_text}"

Please respond with ONLY the JSON object containing the extracted information.
"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a medical data extraction assistant. Extract patient information from natural language and return it as JSON."},
                    {"role": "user", "content": extraction_prompt}
                ],
                max_tokens=2000,
                temperature=0.1
            )
            
            # Parse the JSON response
            extracted_data = json.loads(response.choices[0].message.content)
            
            # Add today's date if not provided
            if extracted_data.get('date') == 'Not mentioned':
                extracted_data['date'] = datetime.now().strftime("%Y-%m-%d")
            
            # Store the original speech text
            extracted_data['original_speech'] = speech_text
            
            print("\n✅ Successfully extracted structured data from speech")
            return extracted_data
            
        except json.JSONDecodeError:
            print("❌ Error parsing extracted data. Using raw speech text.")
            return {
                'name': 'Not extracted',
                'age': 'Not extracted', 
                'gender': 'Not extracted',
                'date': datetime.now().strftime("%Y-%m-%d"),
                'additional_notes': speech_text,
                'original_speech': speech_text
            }
        except Exception as e:
            print(f"❌ Error extracting data: {e}")
            return {
                'name': 'Error in extraction',
                'additional_notes': speech_text,
                'original_speech': speech_text,
                'extraction_error': str(e)
            }
    
    def choose_input_method(self) -> Dict[str, Any]:
        """
        Let user choose between form input and speech input
        
        Returns:
            Dict containing patient data
        """
        print("\n🔄 CHOOSE INPUT METHOD:")
        print("1. 📝 Form-based input (traditional questionnaire)")
        
        if self.speech_enabled:
            print("2. 🎤 Live Microphone input (natural language description)")
            print("   ✅ Live microphone detected and ready for speech recognition")
        else:
            print("2. 📝 Text input (natural language description)")
            print("   ⚠️ Note: Live microphone not available, will use text input for option 2")
        
        while True:
            choice = input("\nEnter your choice (1 or 2): ").strip()
            
            if choice == "1":
                print("\n📝 Using form-based input...")
                return self.collect_patient_data()
            elif choice == "2":
                if self.speech_enabled:
                    print("\n🎤 Using live microphone input...")
                else:
                    print("\n📝 Using text input (microphone not available)...")
                return self.collect_patient_data_via_speech()
            else:
                print("❌ Invalid choice. Please enter 1 or 2.")
    
    def run_flexible_assessment(self) -> str:
        """
        Run assessment with choice of input method
        
        Returns:
            str: Complete assessment including patient data and diagnosis
        """
        print("Starting Fever Proforma Assessment...\n")
        
        # Let user choose input method
        patient_data = self.choose_input_method()
        
        # Format data for LLM
        formatted_data = self.format_patient_data_for_llm(patient_data)
        
        # If we have original speech, include it in the analysis
        if 'original_speech' in patient_data:
            formatted_data += f"\n\nORIGINAL PATIENT DESCRIPTION (from speech):\n{patient_data['original_speech']}"
        
        print("\n" + "="*50)
        print("GENERATING DIAGNOSIS AND NEXT STEPS...")
        print("="*50)
        
        # Generate diagnosis and next steps
        diagnosis_and_steps = self.generate_diagnosis_and_next_steps(formatted_data)
        
        # Combine everything for final output
        final_output = f"""
{formatted_data}

{diagnosis_and_steps}

Assessment completed on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""
        
        return final_output

def main():
    """
    Main function to run the nurse agent
    """
    print("🏥 Welcome to the Fever Proforma Nurse Agent System")
    print("=" * 55)
    print("🚀 Enhanced with Live Microphone Support")
    print("=" * 55)
    
    try:
        # Initialize the nurse agent (API key will be loaded from .env automatically)
        nurse_agent = FeverProformaNurseAgent()
        print("✅ API key loaded successfully from .env file")
        
    except ValueError as e:
        print(f"❌ Error: {e}")
        # Fallback to manual input if .env loading fails
        api_key = input("\nPlease enter your OpenAI API key: ")
        if not api_key:
            print("❌ Error: OpenAI API key is required to proceed.")
            return
        nurse_agent = FeverProformaNurseAgent(api_key)
    
    try:
        # Run the flexible assessment (with choice of input method)
        result = nurse_agent.run_flexible_assessment()
        
        # Display results
        print("\n" + "="*70)
        print("🏥 COMPLETE FEVER PROFORMA ASSESSMENT")
        print("="*70)
        print(result)
        
        # Save to file option
        save_option = input("\n💾 Would you like to save this assessment to a file? (y/n): ")
        if save_option.lower() == 'y':
            filename = f"fever_assessment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            with open(filename, 'w') as f:
                f.write(result)
            print(f"✅ Assessment saved to: {filename}")
        
        print("\n🎉 Assessment completed successfully!")
            
    except KeyboardInterrupt:
        print("\n\n👋 Assessment interrupted by user. Goodbye!")
    except Exception as e:
        print(f"❌ Error during assessment: {str(e)}")

if __name__ == "__main__":
    main()
