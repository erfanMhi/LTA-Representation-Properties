import time
import numpy as np

def run_steps(agent):
    t0 = time.time()
    observations = []
    agent.populate_returns()
    early_model_saved = False
    while True:
        if agent.cfg.log_interval and not agent.total_steps % agent.cfg.log_interval:
            if agent.cfg.tensorboard_logs: agent.log_tensorboard()
            mean, median, min, max = agent.log_file(elapsed_time=agent.cfg.log_interval / (time.time() - t0))
            if agent.cfg.save_params and agent.cfg.save_early is not None and \
                    mean >= agent.cfg.save_early["mean"] and \
                    min >= agent.cfg.save_early["min"] and \
                    (not early_model_saved):
                agent.save(early=True)
                early_model_saved = True
                agent.cfg.logger.info('Save early-stopping model')
            t0 = time.time()

        if agent.cfg.eval_interval and not agent.total_steps % agent.cfg.eval_interval:
            # agent.eval_episodes(elapsed_time=agent.cfg.log_interval / (time.time() - t0))
            # agent.eval_episodes()
            if agent.cfg.visualize and agent.total_steps > 1:
                agent.visualize()
            if agent.cfg.save_params:
                agent.save()
            if agent.cfg.evaluate_lipschitz:
                agent.log_lipschitz()
            t0 = time.time()
        if agent.cfg.max_steps and agent.total_steps >= agent.cfg.max_steps:
            if agent.cfg.save_params:
                agent.save()
            if not early_model_saved and agent.cfg.save_params:
                agent.save(early=True)
                early_model_saved = True
                agent.cfg.logger.info('Save the last learned model as the early-stopping model')
            break
        if agent.cfg.log_observations:
            if agent.reset:
                print(agent.total_steps)
                observations.append([])
            observations[-1].append(agent.state)
        agent.step()
    
    if agent.cfg.log_observations:
        agent.cfg.logger.numpy_log(observations)

def run_steps_nas_study(agent):
    t0 = time.time()
    agent.populate_returns()
    while True:
        if agent.cfg.log_interval and not agent.total_steps % agent.cfg.log_interval:
            if agent.cfg.tensorboard_logs: agent.log_tensorboard()
            agent.log_file(elapsed_time=agent.cfg.log_interval / (time.time() - t0))
            t0 = time.time()
        if agent.cfg.eval_interval and not agent.total_steps % agent.cfg.eval_interval:
            agent.eval_episodes()
            if agent.cfg.visualize:
                agent.visualize()
            if agent.cfg.save_params:
                agent.save()
            if agent.cfg.evaluate_lipschitz:
                agent.log_lipschitz()
            if agent.cfg.evaluate_dynamics_awareness:
                agent.log_dynamics_awareness()
            if agent.cfg.evaluate_decorrelation:
                agent.log_decorrelation()
            if agent.cfg.evaluate_losses:
                agent.log_losses()
            t0 = time.time()
        if agent.cfg.max_steps and agent.total_steps >= agent.cfg.max_steps:
            agent.save()
            break
        agent.step()

def run_modular(agent):
    t0 = time.time()
    agent.cfg.logger.info("Collecting data. Size {}".format(agent.cfg.memory_size))
    while True:
        b_size = agent.random_step()
        if b_size >= agent.cfg.memory_size:
            break
        # if b_size % 100 == 0:
        #     print("Buffer size", b_size)
    agent.cfg.logger.info("Learning")
    while True:
        agent.update_step()
        if agent.cfg.save_interval and not agent.total_steps % agent.cfg.save_interval:
            agent.save()
        if agent.cfg.log_interval and not agent.total_steps % agent.cfg.log_interval:
            if agent.cfg.tensorboard_logs: agent.log_tensorboard()
            agent.log_file(elapsed_time=agent.cfg.log_interval / (time.time() - t0))
            t0 = time.time()
        if agent.cfg.eval_interval and not agent.total_steps % agent.cfg.eval_interval:
            agent.save()
            t0 = time.time()
            if agent.cfg.visualize:
                agent.visualize()
        if agent.cfg.max_steps and agent.total_steps >= agent.cfg.max_steps:
            break

# def run_laplace(agent):
#     t0 = time.time()
#     while True:
#         agent.step()
#         if agent.cfg.save_interval and not agent.total_steps % agent.cfg.save_interval:
#             agent.save()
#         if agent.cfg.log_interval and not agent.total_steps % agent.cfg.log_interval:
#             if agent.cfg.tensorboard_logs: agent.log_tensorboard()
#             agent.log_file(elapsed_time=agent.cfg.log_interval / (time.time() - t0))
#             t0 = time.time()
#         if agent.cfg.eval_interval and not agent.total_steps % agent.cfg.eval_interval:
#             agent.save()
#             t0 = time.time()
#             if agent.cfg.visualize:
#                 agent.visualize()
#         if agent.cfg.max_steps and agent.total_steps >= agent.cfg.max_steps:
#             break


def data_collection_steps(agent):
    while True:
        if agent.check_done():
            break
        agent.step()
        if agent.cfg.log_interval and not agent.total_steps % agent.cfg.log_interval:
            if agent.cfg.tensorboard_logs: agent.log_tensorboard()
            agent.log_file()
