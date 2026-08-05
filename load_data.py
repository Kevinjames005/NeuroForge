import mne

# Load one EEG file (Subject 1, Training session)
raw = mne.io.read_raw_gdf("data/A01T.gdf", preload=True)

# Print basic info about the file
print(raw.info)
print("Channel names:", raw.ch_names)
print("Sampling frequency:", raw.info['sfreq'])

# Extract events (trial markers) from annotations
events, event_dict = mne.events_from_annotations(raw)

print("\nEvent dictionary (label -> code):")
print(event_dict)
print("\nNumber of events found:", len(events))
print("First 5 events:\n", events[:5])

