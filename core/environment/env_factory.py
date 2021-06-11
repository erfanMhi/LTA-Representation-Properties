import os

# from core.environment.acrobot import Acrobot
from core.environment.gridworlds_goal import GridHardXYGoal, GridHardRGBGoal
from core.environment.gridworlds_noise import *
from core.environment.collectworlds import CollectRGB, CollectColor, CollectTwoColorRGB, CollectTwoColor, CollectTwoColorLip, \
    CollectTwoColorRGBFix, CollectTwoColorRGBSimple, CollectRandColorRGB, CollectRandColorRGBTest
from core.environment.mountaincar import *
from core.environment.collectworlds_random import CollectRandomRGB


class EnvFactory:
    @classmethod
    def create_env_fn(cls, cfg):
        # if cfg.env_name == 'Acrobot':
        #     return lambda: Acrobot(cfg.seed)
        # elif cfg.env_name == 'GridHardXY':
        if cfg.env_name == 'GridHardXY':
            return lambda: GridHardXY(cfg.seed)
        elif cfg.env_name == 'GridHardGS':
            return lambda: GridHardGS(cfg.seed)
        elif cfg.env_name == 'GridHardRGB':
            return lambda: GridHardRGB(cfg.seed)
        elif cfg.env_name == 'GridHardXYGoal':
            return lambda: GridHardXYGoal(cfg.goal_id, cfg.seed)
        elif cfg.env_name == 'GridHardRGBGoal':
            return lambda: GridHardRGBGoal(cfg.goal_id, cfg.seed)
        elif cfg.env_name == 'NoisyGridHardXY':
            return lambda: NoisyGridHardXY(cfg.seed)
        elif cfg.env_name == 'NoisyGridHardGS':
            return lambda: NoisyGridHardGS(cfg.seed)
        elif cfg.env_name == 'CollectRGB':
            return lambda: CollectRGB(cfg.seed)
        elif cfg.env_name == "CollectTwoColorRGB":
            return lambda: CollectTwoColorRGB(cfg.seed, fruit_num=cfg.fruit_num, single_channel_color=cfg.single_channel_color, rewarding_color=cfg.rewarding_color)
        elif cfg.env_name == "CollectRandColorRGB":
            return lambda: CollectRandColorRGB(cfg.seed, fruit_num=cfg.fruit_num)
        elif cfg.env_name == "CollectRandColorRGBTest":
            return lambda: CollectRandColorRGBTest(cfg.seed, fruit_num=cfg.fruit_num, color=cfg.test_rewarding_color)
        elif cfg.env_name == "CollectTwoColorRGBFix":
            return lambda: CollectTwoColorRGBFix(cfg.seed)
        elif cfg.env_name == "CollectTwoColorRGBSimple":
            return lambda: CollectTwoColorRGBSimple(cfg.seed)
        elif cfg.env_name == 'CollectColor':
            return lambda: CollectColor(cfg.rewarding_color, cfg.seed)
        elif cfg.env_name == 'CollectTwoColor':
            return lambda: CollectTwoColor(cfg.rewarding_color, cfg.seed, fruit_num=cfg.fruit_num)
        elif cfg.env_name == 'CollectTwoColorLip':
            lip_sampled_states_path = os.path.join(cfg.data_root, cfg.lipschitz_sampled_states_path)
            return lambda: CollectTwoColorLip(lip_sampled_states_path, cfg.rewarding_color, cfg.seed, fruit_num=cfg.fruit_num)
        elif cfg.env_name == 'CollectRandomRGB':
            return lambda: CollectRandomRGB(cfg.seed)
        elif cfg.env_name == 'GridTwoRoomRGB':
            return lambda: GridTwoRoomRGB(cfg.seed)
        elif cfg.env_name == 'GridOneRoomRGB':
            return lambda: GridOneRoomRGB(cfg.seed)
        elif cfg.env_name == 'FakeAgentGridHardRGBSimple':
            return lambda: grid_fake_agent_simple("GridHardRGB", cfg.seed)
        elif cfg.env_name == 'FakeAgentGridTwoRoomRGBSimple':
            return lambda: grid_fake_agent_simple("GridTwoRoomRGB", cfg.seed)
        elif cfg.env_name == 'FakeAgentGridOneRoomRGBSimple':
            return lambda: grid_fake_agent_simple("GridOneRoomRGB", cfg.seed)
        elif cfg.env_name == 'FakeAgentGridHardRGB':
            return lambda: grid_fake_agent("GridHardRGB", cfg.seed)
        elif cfg.env_name == 'FakeAgentGridTwoRoomRGB':
            return lambda: grid_fake_agent("GridTwoRoomRGB", cfg.seed)
        elif cfg.env_name == 'FakeAgentGridOneRoomRGB':
            return lambda: grid_fake_agent("GridOneRoomRGB", cfg.seed)
        elif cfg.env_name == 'NumberGridHardRGB':
            return lambda: grid_number("GridHardRGB", cfg.seed)
        elif cfg.env_name == 'NumberGridTwoRoomRGB':
            return lambda: grid_number("GridTwoRoomRGB", cfg.seed)
        elif cfg.env_name == 'NumberGridOneRoomRGB':
            return lambda: grid_number("GridOneRoomRGB", cfg.seed)
        elif cfg.env_name == 'BackgroundGridHardRGB':
            return lambda: grid_background("GridHardRGB", cfg.seed, cfg.change_pxl)
        elif cfg.env_name == 'BackgroundGridTwoRoomRGB':
            return lambda: grid_background("GridTwoRoomRGB", cfg.seed, cfg.change_pxl)
        elif cfg.env_name == 'BackgroundGridOneRoomRGB':
            return lambda: grid_background("GridOneRoomRGB", cfg.seed, cfg.change_pxl)
        elif cfg.env_name == 'MountainCar':
            return lambda: MountainCar(cfg.seed)

        # elif cfg.env_name in ['FakeAgentGridHardRGBSimple', 'FakeAgentGridTwoRoomRGBSimple', 'FakeAgentGridOneRoomRGBSimple']:
        #     return lambda: grid_fake_agent_simple(cfg.env_name, cfg.seed)
        # elif cfg.env_name in ['FakeAgentGridHardRGB', 'FakeAgentGridTwoRoomRGB', 'FakeAgentGridOneRoomRGB']:
        #     return lambda: grid_fake_agent(cfg.env_name, cfg.seed)
        # elif cfg.env_name in ['NumberGridHardRGB', 'NumberGridTwoRoomRGB', 'NumberGridOneRoomRGB']:
        #     return lambda: grid_number(cfg.env_name, cfg.seed)
        # elif cfg.env_name in ['BackgroundGridHardRGB', 'BackgroundGridTwoRoomRGB', 'BackgroundGridOneRoomRGB']:
        #     return lambda: grid_background(cfg.env_name, cfg.seed, cfg.change_pxl)

        # elif cfg.env_name == 'FakeAgentGridHardRGBSimple':
        #     return lambda: FakeAgentGridHardRGBSimple(cfg.seed)
        # elif cfg.env_name == 'FakeAgentGridHardRGB':
        #     return lambda: FakeAgentGridHardRGB(cfg.seed)
        # elif cfg.env_name == 'NumberGridHardRGB':
        #     return lambda: NumberGridHardRGB(cfg.seed)
        # elif cfg.env_name == 'BackgroundGridHardRGB':
        #     return lambda: BackgroundGridHardRGB(cfg.seed, cfg.change_pxl)

        else:
            print(cfg.env_name)
            raise NotImplementedError




