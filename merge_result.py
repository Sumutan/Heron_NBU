import sys
from src.metrics import merge, merge_notebook

# final_top1 ,final_top5 = merge('./output', int(sys.argv[1]))
# print(f'final_top1: {final_top1}, final_top5: {final_top5}')

final_top1, final_top5 = merge_notebook('./device/output', 8)
print(f'final_top1: {final_top1}, final_top5: {final_top5}')


