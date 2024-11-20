from pettingzoo.utils.wrappers.order_enforcing import OrderEnforcingWrapper

class AECWrapper(OrderEnforcingWrapper):
    def __init__(self, env):
        super().__init__(env.env)
        
    def observation_space(self, agent):
        # TODO
        return super().observation_space(agent)
    
    def observe(self, agent):
        # TODO
        return super().observe(agent)