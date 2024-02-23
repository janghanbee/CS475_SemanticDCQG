import json
from argparse import ArgumentParser
from metric.text_generation_metrics import compute_metrics_by_file
parser = ArgumentParser()
parser.add_argument("--eval_result_output", type=str, default="", help="eval_result_output")
parser.add_argument("--output_prefix", type=str, default="", help="output_prefix")
parser.add_argument("--filename", type=str, default="", help="filename")
args = parser.parse_args()

output_prefix = args.output_prefix
filename = args.filename
eval_result_output = arg.eval_result_output

with open(output_prefix+filename,'r', encoding='utf-8') as file:
    data = json.load(file)

gold_file = output_prefix + "gold_gpt2gold_gpt2.txt"
pred_file = output_prefix + "pred_gpt2.txt"
fgold = open(gold_file, "w", encoding="utf-8")
fpred = open(pred_file, "w", encoding="utf-8")
data_to_write = data["data"][0]['paragraphs']
for d in data_to_write:
    fgold.write(d["qas"][0]["original_question"].rstrip() + "\n")
    fpred.write(d["qas"][0]["question"].rstrip() + "\n")
fgold.close()
fpred.close()

ret_scores = compute_metrics_by_file([gold_file], pred_file)
print(ret_scores)
fres = open(eval_result_output, "w", encoding="utf-8")
fres.write(str(ret_scores))