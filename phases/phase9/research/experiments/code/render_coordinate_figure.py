from pathlib import Path
import os,json
root=Path(__file__).resolve().parents[1]
os.environ['MPLCONFIGDIR']=str(root/'logs/matplotlib_cache')
os.environ['XDG_CACHE_HOME']=str(root/'logs/cache')
(root/'logs/cache').mkdir(parents=True,exist_ok=True)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
rows=json.loads((root/'results/coordinate_obstruction.json').read_text())['rows']
r=[x['curvature_degree'] for x in rows]
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
fig,ax=plt.subplots(1,2,figsize=(9,3.8))
ax[0].loglog(r,[x['partition_product_lower'] for x in rows],'o-',color='#176b86',label='Two-partition comparison product')
ax[0].loglog(r,[1.]*len(rows),'s--',color='#ba5328',label='Actual hierarchy price = 1')
ax[0].set(xlabel='Curvature degree r = 4m − 4',ylabel='Dimensionless factor',title='A limitation of the comparison method')
ax[0].legend(loc='upper left',fontsize=8)
ax[1].loglog(r,[x['normalized_OPT2'] for x in rows],'o-',color='#6b4b8b')
ax[1].set(xlabel='Curvature degree r = 4m − 4',ylabel='Normalized flat excess distortion',title='Absolute costs vanish in these instances')
for a in ax:a.grid(True,alpha=.25,which='major')
fig.tight_layout(rect=(0,.10,1,1))
fig.text(.5,.02,'Three states at 0, 1/2, 1; weights ∝ (1, η, η²), η = 10⁻³/m⁴. Curvature divided by m⁴ + m².\nThe comparison-product curve is a lower bound from two partitions, not a hierarchy-price lower bound.',ha='center',fontsize=8)
(root/'figures').mkdir(exist_ok=True)
fig.savefig(root/'figures/coordinate_method_obstruction.pdf',bbox_inches='tight')
fig.savefig(root/'figures/coordinate_method_obstruction.png',dpi=180,bbox_inches='tight')
print('coordinate figure rendered')
