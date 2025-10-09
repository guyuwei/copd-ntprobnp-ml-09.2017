import ssl

import pandas as pd

ssl._create_default_https_context = ssl._create_unverified_context

featuresall = ['HTN', 'DM', 'CHD', 'Gender', 'smoking index', 'BMI', 'COPD course', 'FEV1/FVC%', 'FEV1%Pred', 'FVC%Pred', 'DLCO%pred', 'Cl', 'hs-cTn', 'D-D', 'PH', 'OI', 'PCO2', 'NEU%', 'HGB', 'PLT',
               'PASP', 'FEV1-BEST']

features = ['Age', 'smoking index', 'FEV1%Pred', 'AST', 'hs-cTn',
            'PCO2', 'CRP', 'HGB', ]

train = pd.read_csv( "/Users/gyw/Desktop/Project/慢阻肺合并症-gyw/02Trainset.csv" )

X_train = train[features]
corr = X_train[features].corr( )
y_train = train["PAHD"]
# print( d )

print( (len( features ), features) )
list = ['Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BuGn', 'BuGn_r', 'BuPu', 'BuPu_r', 'CMRmap', 'CMRmap_r', 'Dark2', 'Dark2_r', 'GnBu', 'GnBu_r', 'Greens', 'Greens_r', 'Greys',
        'Greys_r', 'OrRd', 'OrRd_r', 'Oranges', 'Oranges_r', 'PRGn', 'PRGn_r', 'Paired', 'Paired_r', 'Pastel1', 'Pastel1_r', 'Pastel2', 'Pastel2_r', 'PiYG', 'PiYG_r', 'PuBu', 'PuBuGn', 'PuBuGn_r',
        'PuBu_r', 'PuOr', 'PuOr_r', 'PuRd', 'PuRd_r', 'Purples', 'Purples_r', 'RdBu', 'RdBu_r', 'RdGy', 'RdGy_r', 'RdPu',
        'RdPu_r', 'Reds', 'Reds_r',
        'Set1', 'Set1_r', 'Set2', 'Set2_r', 'Set3', 'Set3_r', 'Spectral', 'Spectral_r', 'Wistia', 'Wistia_r',
        'afmhot', 'afmhot_r', 'autumn', 'autumn_r', 'binary', 'binary_r', 'bone', 'bone_r', 'brg', 'brg_r', 'bwr', 'bwr_r', 'cividis', 'cividis_r', 'cool', 'cool_r', 'coolwarm', 'coolwarm_r',
        'copper', 'copper_r', 'cubehelix', 'cubehelix_r', 'flag', 'flag_r', 'gist_earth', 'gist_earth_r', 'gist_gray', 'gist_gray_r', 'gist_heat', 'gist_heat_r', 'gist_ncar', 'gist_ncar_r',
        'gist_rainbow', 'gist_rainbow_r', 'gist_stern', 'gist_stern_r', 'gist_yarg', 'gist_yarg_r', 'gnuplot', 'gnuplot2', 'gnuplot2_r', 'gnuplot_r', 'gray', 'gray_r', 'hot', 'hot_r', 'hsv', 'hsv_r',
        'icefire', 'icefire_r', 'inferno', 'inferno_r', 'jet', 'jet_r', 'magma', 'magma_r', 'mako', 'mako_r', 'nipy_spectral', 'nipy_spectral_r', 'ocean', 'ocean_r', 'pink', 'pink_r', 'plasma',
        'plasma_r', 'prism', 'prism_r', 'rainbow', 'rainbow_r', 'rocket', 'rocket_r', 'seismic', 'seismic_r', 'spring', 'spring_r', 'summer', 'summer_r', 'tab10', 'tab10_r', 'tab20', 'tab20_r',
        'tab20b', 'tab20b_r', 'tab20c', 'tab20c_r', 'terrain', 'terrain_r', 'twilight', 'twilight_r', 'twilight_shifted', 'twilight_shifted_r', 'viridis', 'viridis_r', 'vlag', 'vlag_r', 'winter',
        'winter_r']
list = ['PiYG', 'PiYG_r', 'PuBu', 'PuBuGn', 'PuBuGn_r',
        'PuBu_r', 'PuOr', 'PuOr_r', 'PuRd', 'PuRd_r', 'Purples', 'Purples_r', 'RdBu', 'RdBu_r', 'RdGy', 'RdGy_r', 'RdPu', ]
# for i in list:
# 	print(i)
# 	plt.subplots( figsize = (15,15), dpi = 300 )
# 	sns.heatmap( corr, cmap = i, annot = True, linewidths = 0.05, linecolor = "grey", annot_kws = { 'size': 20 } )
# 	sns.set(font_scale =3)
# 	# plt.savefig( "/Users/gyw/Desktop/Project/慢阻肺合并症-gyw/热力图/" + i + ".png" )
# 	plt.tight_layout()
# 	plt.show( )