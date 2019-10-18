o_km = json.load(open('kmeans_average_family_adj_multual_info.json', 'rb'))
o_kn = json.load(open('knn_average_family_adj_multual_info.json', 'rb'))
for family in families: 
    print(family, ' & {:.3f} & {:.3f} & {:.3f} & {:.3f} & {:.3f} & {:.3f} & {:.3f} & {:.3f} & {:.3f} & {:.3f} \\'.format(
            o_km[family]['mfcc_mean'], o_kn[family]['mfcc_mean'], 
            o_km[family]['audioset_mean'], o_kn[family]['audioset_mean'], 
            o_km[family]['openl3-music_mean'], o_kn[family]['openl3-music_mean'], 
            o_km[family]['openl3-env_mean'], o_kn[family]['openl3-env_mean'], 
            o_km[family]['soundnet_mean'], o_kn[family]['soundnet_mean'], 
        )
    )
print('Average', ' & {:.3f} & {:.3f} & {:.3f} & {:.3f} & {:.3f} & {:.3f} & {:.3f} & {:.3f} & {:.3f} & {:.3f} \\'.format( 
        np.mean([o_km[family]['mfcc_mean'] for family in families]), np.mean([o_kn[family]['mfcc_mean'] for family in families]),
        np.mean([o_km[family]['audioset_mean'] for family in families]), np.mean([o_kn[family]['audioset_mean'] for family in families]),
        np.mean([o_km[family]['openl3-music_mean'] for family in families]), np.mean([o_kn[family]['openl3-music_mean'] for family in families]),
        np.mean([o_km[family]['openl3-env_mean'] for family in families]), np.mean([o_kn[family]['openl3-env_mean'] for family in families]),
        np.mean([o_km[family]['soundnet_mean'] for family in families]), np.mean([o_kn[family]['soundnet_mean'] for family in families])
    )
)
