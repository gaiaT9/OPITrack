import config_mots.car_test_se_to_save
import config_mots.car_test_tracking_val
import config_mots.car_finetune_tracking
import config_mots.car_finetune_SE_crop
import config_mots.car_finetune_SE_mots
import config_mots.car_test_se_to_save_ours
import config_mots.car_test_se_to_save_testset
import config_mots.car_test_tracking_test
# point track pp
import config_mots.car_finetune_tracking_pp
import config_mots.car_test_tracking_val_pp

import config_mots.person_finetune_tracking
import config_mots.person_test_tracking_val
# additional config
import config_mots.car_test_se_to_save_trainset
import config_mots.car_test_tracking_train

# auto import 
import os
current_files = os.listdir(os.path.dirname(__file__))
for f in current_files:
    if f.find('.py') != -1 and f.find('__') == -1:
        exec('import config_mots.%s' % (f.split('.py')[0]))