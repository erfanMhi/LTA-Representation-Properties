import os

# from core.environment.acrobot import Acrobot
from core.environment.gridworlds_goal import GridHardXYGoal, GridHardRGBGoal
from core.environment.gridworlds_noise import *
from core.environment.collectworlds import CollectRGB, CollectColor, CollectTwoColorRGB, CollectTwoColor, CollectTwoColorLip, \
    CollectTwoColorRGBFix, CollectTwoColorRGBSimple
from core.environment.collectworlds_random import CollectRandomRGB


class EnvFactory:
    @classmethod
    def create_env_fn(cls, cfg):
        # if cfg.env_name == 'Acrobot':
        #     return lambda: Acrobot(cfg.id)
        # elif cfg.env_name == 'GridHardXY':
        if cfg.env_name == 'GridHardXY':
            return lambda: GridHardXY(cfg.id)
        elif cfg.env_name == 'GridHardGS':
            return lambda: GridHardGS(cfg.id)
        elif cfg.env_name == 'GridHardRGB':
            return lambda: GridHardRGB(cfg.id)
        elif cfg.env_name == 'GridHardXYGoal':
            return lambda: GridHardXYGoal(cfg.goal_id, cfg.id)
        elif cfg.env_name == 'GridHardRGBGoal':
            return lambda: GridHardRGBGoal(cfg.goal_id, cfg.id)
        elif cfg.env_name == 'NoisyGridHardXY':
            return lambda: NoisyGridHardXY(cfg.id)
        elif cfg.env_name == 'NoisyGridHardGS':
            return lambda: NoisyGridHardGS(cfg.id)
        elif cfg.env_name == 'CollectRGB':
            return lambda: CollectRGB(cfg.id)
        elif cfg.env_name == "CollectTwoColorRGB":
            return lambda: CollectTwoColorRGB(cfg.id)
        elif cfg.env_name == "CollectTwoColorRGBFix":
            return lambda: CollectTwoColorRGBFix(cfg.id)
        elif cfg.env_name == "CollectTwoColorRGBSimple":
            return lambda: CollectTwoColorRGBSimple(cfg.id)
        elif cfg.env_name == 'CollectColor':
            return lambda: CollectColor(cfg.rewarding_color, cfg.id)
        elif cfg.env_name == 'CollectTwoColor':
            return lambda: CollectTwoColor(cfg.rewarding_color, cfg.id)
        elif cfg.env_name == 'CollectTwoColorLip':
            lip_sampled_states_path = os.path.join(cfg.data_root, cfg.lipschitz_sampled_states_path)
            return lambda: CollectTwoColorLip(lip_sampled_states_path, cfg.rewarding_color, cfg.id)
        elif cfg.env_name == 'CollectRandomRGB':
            return lambda: CollectRandomRGB(cfg.id)
        elif cfg.env_name == 'GridTwoRoomRGB':
            return lambda: GridTwoRoomRGB(cfg.id)
        elif cfg.env_name == 'GridOneRoomRGB':
            return lambda: GridOneRoomRGB(cfg.id)
        elif cfg.env_name == 'FakeAgentGridHardRGBSimple':
            return lambda: grid_fake_agent_simple("GridHardRGB", cfg.id)
        elif cfg.env_name == 'FakeAgentGridTwoRoomRGBSimple':
            return lambda: grid_fake_agent_simple("GridTwoRoomRGB", cfg.id)
        elif cfg.env_name == 'FakeAgentGridOneRoomRGBSimple':
            return lambda: grid_fake_agent_simple("GridOneRoomRGB", cfg.id)
        elif cfg.env_name == 'FakeAgentGridHardRGB':
            return lambda: grid_fake_agent("GridHardRGB", cfg.id)
        elif cfg.env_name == 'FakeAgentGridTwoRoomRGB':
            return lambda: grid_fake_agent("GridTwoRoomRGB", cfg.id)
        elif cfg.env_name == 'FakeAgentGridOneRoomRGB':
            return lambda: grid_fake_agent("GridOneRoomRGB", cfg.id)
        elif cfg.env_name == 'NumberGridHardRGB':
            return lambda: grid_number("GridHardRGB", cfg.id)
        elif cfg.env_name == 'NumberGridTwoRoomRGB':
            return lambda: grid_number("GridTwoRoomRGB", cfg.id)
        elif cfg.env_name == 'NumberGridOneRoomRGB':
            return lambda: grid_number("GridOneRoomRGB", cfg.id)
        elif cfg.env_name == 'BackgroundGridHardRGB':
            return lambda: grid_background("GridHardRGB", cfg.id, cfg.change_pxl)
        elif cfg.env_name == 'BackgroundGridTwoRoomRGB':
            return lambda: grid_background("GridTwoRoomRGB", cfg.id, cfg.change_pxl)
        elif cfg.env_name == 'BackgroundGridOneRoomRGB':
            return lambda: grid_background("GridOneRoomRGB", cfg.id, cfg.change_pxl)

        # elif cfg.env_name in ['FakeAgentGridHardRGBSimple', 'FakeAgentGridTwoRoomRGBSimple', 'FakeAgentGridOneRoomRGBSimple']:
        #     return lambda: grid_fake_agent_simple(cfg.env_name, cfg.id)
        # elif cfg.env_name in ['FakeAgentGridHardRGB', 'FakeAgentGridTwoRoomRGB', 'FakeAgentGridOneRoomRGB']:
        #     return lambda: grid_fake_agent(cfg.env_name, cfg.id)
        # elif cfg.env_name in ['NumberGridHardRGB', 'NumberGridTwoRoomRGB', 'NumberGridOneRoomRGB']:
        #     return lambda: grid_number(cfg.env_name, cfg.id)
        # elif cfg.env_name in ['BackgroundGridHardRGB', 'BackgroundGridTwoRoomRGB', 'BackgroundGridOneRoomRGB']:
        #     return lambda: grid_background(cfg.env_name, cfg.id, cfg.change_pxl)

        # elif cfg.env_name == 'FakeAgentGridHardRGBSimple':
        #     return lambda: FakeAgentGridHardRGBSimple(cfg.id)
        # elif cfg.env_name == 'FakeAgentGridHardRGB':
        #     return lambda: FakeAgentGridHardRGB(cfg.id)
        # elif cfg.env_name == 'NumberGridHardRGB':
        #     return lambda: NumberGridHardRGB(cfg.id)
        # elif cfg.env_name == 'BackgroundGridHardRGB':
        #     return lambda: BackgroundGridHardRGB(cfg.id, cfg.change_pxl)

        else:
            print(cfg.env_name)
            raise NotImplementedError




