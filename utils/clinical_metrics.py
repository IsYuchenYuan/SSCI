import scipy.stats as stats
def cal_EF():
    ed_vol = np.load(test_save_path+'ED_vol.npy')
    es_vol = np.load(test_save_path+'ES_vol.npy')
    ed_err = np.load(test_save_path+'ED_err.npy')
    es_err = np.load(test_save_path+'ES_err.npy')
    for struc_name in [0, 2]:
        ED_vol = ed_vol[:,struc_name]
        ES_vol = es_vol[:,struc_name]
        EF_pred = (ED_vol - ES_vol) / ED_vol

        ED_vol_gt = ED_vol - ed_err[:,struc_name]
        ES_vol_gt = ES_vol - es_err[:,struc_name]

        EF_gt = (ED_vol_gt - ES_vol_gt) / ED_vol_gt

        LV_EF_corr = stats.pearsonr(EF_pred, EF_gt)[0]
        LV_EF_corr_r2 = r2_score(EF_pred, EF_gt)

        # EF_err = mean_squared_error(EF_pred, EF_gt, squared=False)
        print('{}, EF corr: {}\n\n'.format(struc_name, LV_EF_corr))
        print('{}, EF r2 corr: {}\n\n'.format(struc_name, LV_EF_corr_r2))
        # print('{}, EF err: {}\n\n'.format(struc_name, EF_err))


# vol_list = np.reshape(vol_list, [-1, 3])
    # volume_err = np.reshape(volume_err, [-1, 3])
    # vol_gt = np.reshape(vol_gt, [-1, 3])
    #
    # np.save(test_save_path+ _phase +'_vol.npy',vol_list)
    # np.save(test_save_path+ _phase + '_err.npy',volume_err)
    #
    #
    # logging.info('Volume MAE:')
    # logging.info('RV :%.3f' % (mean_absolute_error(vol_list[:,0], vol_gt[:,0])))
    # logging.info('MYO :%.3f' % (mean_absolute_error(vol_list[:,1], vol_gt[:,1])))
    # logging.info('LV :%.3f' % (mean_absolute_error(vol_list[:,2], vol_gt[:,2])))
    #
    # logging.info('Volume corr:')
    # logging.info('RV :%.3f' % ((r2_score(vol_list[:,0], vol_gt[:,0]))))
    # logging.info('MYO :%.3f' % ((r2_score(vol_list[:,1], vol_gt[:,1]))))
    # logging.info('LV :%.3f' % ((r2_score(vol_list[:,2], vol_gt[:,2]))))
    #
    #
    # volume_err_mean = np.mean(volume_err, axis=0)
    # volume_err_std = np.std(volume_err, axis=0)
    #
    # logging.info('Volume error:')
    # logging.info('RV :%.3f(%.3f)' % (volume_err_mean[0], volume_err_std[0]))
    # logging.info('MYO :%.3f(%.3f)' % (volume_err_mean[1], volume_err_std[1]))
    # logging.info('LV :%.3f(%.3f)' % (volume_err_mean[2], volume_err_std[2]))
    # logging.info('dice mean :%.3f' % (np.mean(volume_err_mean)))


# Compute volume
                # if c==2 and phase=='ED':
                #     # MYO -> (g)
                #     volpred = pred_test_data_tr.sum() * np.prod(img_dat[2].get_zooms()) / 1000. * 1.05
                #     volgt = pred_gt_data_tr.sum() * np.prod(img_dat[2].get_zooms()) / 1000. * 1.05
                # else:
                #     volpred = pred_test_data_tr.sum() * np.prod(img_dat[2].get_zooms()) / 1000.
                #     volgt = pred_gt_data_tr.sum() * np.prod(img_dat[2].get_zooms()) / 1000.
                #
                # vol_gt.append(volgt)
                # vol_list.append(volpred)
                # volume_err.append(volpred - volgt)