
import json

def find_and_apply_threshold():
    input_file = 'runs/12_12_llada_8b_eli5_wm_test/gen_length_400/results.jsonl'
    output_file = 'runs/12_12_llada_8b_eli5_wm_test/gen_length_400/results.jsonl.updated'

    # Step 1: Read all scores for non-watermarked text
    no_wm_scores = []
    with open(input_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            if 'detection_no_wm' in data and 'score' in data['detection_no_wm']:
                no_wm_scores.append(data['detection_no_wm']['score'])

    # Step 2: Find the 10th smallest score
    no_wm_scores.sort()
    if len(no_wm_scores) >= 10:
        new_threshold = no_wm_scores[9]
    else:
        print("Not enough samples to determine a new threshold. Found only {} samples.".format(len(no_wm_scores)))
        return

    print(f"New threshold determined: {new_threshold}")

    # Step 3: Read the file again and apply the new threshold
    updated_lines = []
    with open(input_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            
            # Update watermark detection results
            for key in ['detection_wm', 'detection_no_wm']:
                if key in data:
                    data[key]['is_watermarked'] = data[key]['score'] < new_threshold
                    data[key]['threshold'] = new_threshold
                    if 'original_threshold' in data[key]:
                        del data[key]['original_threshold']

            updated_lines.append(json.dumps(data))

    # Step 4: Write the updated data to a new file
    with open(output_file, 'w') as f:
        for line in updated_lines:
            f.write(line + '\n')
            
    print(f"Updated results written to {output_file}")
    
    # Step 5: As the user requested to modify the results.jsonl file, 
    # I'll rename the updated file to the original filename.
    import os
    os.rename(output_file, input_file)
    print(f"Renamed {output_file} to {input_file}")

if __name__ == '__main__':
    find_and_apply_threshold()
