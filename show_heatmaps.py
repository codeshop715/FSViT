import torch
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
import os

# data from train and val
l1 = [445, 348, 748, 1050, 1756, 5130, 8459, 11446, 14053, 18855, 39227, 122691, 121018, 33524]
l2 = [525,740, 1072, 2341, 3767, 9712, 18106, 24043, 28024, 39924, 84975, 238048, 236149,  66324]
l3 = [799, 904, 1469, 3546, 5901, 14557, 27077, 35477, 40804, 59982, 127875, 355054, 354877, 100428]
l4 = [2050,1584, 2138, 4681, 8962, 19945, 36772, 48137, 54598, 81425, 173775, 466258, 469635,  133790]
l5 = [2623, 2186, 2733, 5943, 12465, 24620, 44736, 58870, 68624, 100949, 216337, 582081, 589065, 167518]
l6 = [3093,2918, 4163, 8102, 16288, 29412, 52941, 70215, 82419, 119200, 257679, 699122, 706393,  201805]
l7 = [3493, 3886, 4697, 9945, 22133, 38231, 68171, 89312, 107870, 158892, 344595, 938749, 944588, 269188]
l8 = [3999,5341, 5805, 11482, 25484, 43588, 77574, 100361, 118576, 173946, 382723, 1056981, 1068983,  303907]
l9 = [4291,5510, 6208, 13586, 29454, 50324, 87515, 112269, 132795, 193736, 424063, 1172711, 1184238,  337050]
l10 =[4299, 5522, 6298, 13608, 29579, 50508, 88416, 113359, 134157, 195192, 428053, 1185256, 1196560, 340443]
l = []
l.append(np.array(l1)*100.0/np.array(l1).sum())
l.append((np.array(l2)-np.array(l1))*100.0/(np.array(l1).sum()))
l.append((np.array(l3)-np.array(l2))*100.0/(np.array(l3)-np.array(l2)).sum())
l.append((np.array(l4)-np.array(l3))*100.0/(np.array(l4)-np.array(l3)).sum())
l.append((np.array(l5)-np.array(l4))*100.0/(np.array(l5)-np.array(l4)).sum())
l.append((np.array(l6)-np.array(l5))*100.0/(np.array(l6)-np.array(l5)).sum())
l.append((np.array(l7)-np.array(l6))*100.0/(np.array(l7)-np.array(l6)).sum())
l.append((np.array(l8)-np.array(l7))*100.0/(np.array(l8)-np.array(l7)).sum())
l.append((np.array(l9)-np.array(l8))*100.0/(np.array(l9)-np.array(l8)).sum())
l.append((np.array(l10)-np.array(l9))*100.0/(np.array(l10)-np.array(l9)).sum())
l = np.array(l)

sns.set_context({"figure.figsize":(12,12)})
sns.set(font_scale=0.8)
sns.heatmap(l,cmap="crest",annot=True, fmt=".1f",linewidths=0.5,square=True, cbar=True,cbar_kws={"shrink": 0.5},
            xticklabels=['1×1','2×2','3×3','4×4','5×5','6×6','7×7','8×8','9×9','10×10','11×11','12×12','13×13','14×14',], 
            yticklabels=['baby','bed','bicycle','chimpanzee','fox','leopard','man','pickup_truck','plain','poppy',])
figure_save_path = "./PMSF"
plt.xticks(fontsize=12  # Font size
           , rotation=45  # Whether the font is rotated
           , horizontalalignment='center'  # The relative position of the scale
           )
plt.yticks(fontsize=12)
plt.title('Top-10 percentage of similarity', fontsize=15)
plt.savefig(os.path.join(figure_save_path , '1'))