import json
import openai
from Src.config import openai_api_key
from typing import List, Dict, Optional
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

def process_sample(sample:Dict) -> Dict:
    new_sample = {}
    for key in sample.keys():
        new_sample[key] = sample[key]
    new_sample["subject"] = extract_subject(new_sample["base_prompt"])
    # compute subject position
    return new_sample

def extract_subject(prompt:str) -> str:
    IN_CONTEXT_PROMPT = f'''I will give you a sentene and you should return me the subject. For example:
    sentence: "Seattle City Light is based in" subject:"Seattle City Light"
    sentence: "Italian Empire's capital city," subject: "Italian Empire",
    sentence: "The official language of Ch\u00e2tillon is" subject: " Ch\u00e2tillon" \n "" 
    sentence: "{prompt}" subject:'''
    openai.api_key = openai_api_key
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo-0125",  #! TODO: set temperature to 0    use a specifi checkpoint of the model
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": IN_CONTEXT_PROMPT},
        ],
        temperature=0,
    )
    string_response = response.choices[0].message.content
    #process the string response to get the subject
    # string_response = string_response.split('"')[1].split('"')[0]
    return string_response

class AutoSubjectGenerator:
    def __init__(self, path:str, openai_model:str = "gpt-3.5-turbo-1106"):
        self.path = path
        self.openai_model = openai_model
        self.original_data = self.load_original_data()

        
    def load_original_data(self):
        with open(self.path, "r") as f:
            data = json.load(f)
        #checke the structure of the data from a template
        template_keys = ["base_prompt", "template", "target_true", "target_new", "prompt"]
        # check if the data has the right structure
        for sample in data:
            for key in template_keys:
                assert key in sample.keys(), f"Found a sample without the key {key}"
            if "subject" in sample.keys():
                raise ValueError("The data already has a subject key")
        return data
    
    def process_data(self):
        new_data = []
        self.new_dataset, index = self.load_checkpoints()
        print("Using", cpu_count(), " CPU to make multiple API call to OpenAI")
        with Pool(4) as p:
            with tqdm(total=len(self.original_data[index:]), desc="Extracting subjects from prompts") as pbar:
                for processed_sample in p.imap(process_sample, self.original_data[index:]):
                    pbar.update()
                    new_data.append(processed_sample)

                    # Checkpoint saving
                    #just one thread should save the checkpoint 
                    # put a barrier to avoid multiple threads saving the checkpoint
                    
                    if pbar.n % 100 == 0:
                        self.save_dataset(new_data[:pbar.n], checkpoint=True)
        return new_data
    
    def save_dataset(self, data:List[Dict], checkpoint:bool = False):
        if checkpoint:
            print(f"Saving checkpoint to {self.path}_checkpoint.json with {len(data)} samples processed so far out of {len(self.original_data)}")
            with open(f"{self.path}_checkpoint.json", "w") as f:
                json.dump(data, f, indent=4)
        else:
            with open(f"{self.path}_with_subjects.json", "w") as f:
                json.dump(data, f, indent=4)
                
    
    def run(self):
        dataset = self.process_data()
        self.save_dataset(dataset)
        
    def load_checkpoints(self):
        try:
            with open(f"{self.path}_checkpoint.json", "r") as f:
                return json.load(f), len(json.load(f))
        except:
            return self.original_data, 0
        
    
        
        