class OnPolicyRunner:
    """On-policy runner 簡化版 - 僅保留推論相關功能"""

    def __init__(self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device: str = "cpu") -> None:
        self.cfg = train_cfg
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env

        obs = self.env.get_observations()
        self.cfg["obs_groups"] = resolve_obs_groups(obs, self.cfg["obs_groups"], ["critic"])

        # 核心：構建神經網路結構
        self.alg = self._construct_algorithm(obs)

    def load(self, path: str, load_optimizer: bool = False, map_location: str | None = None) -> dict:
        """加載模型權重"""
        loaded_dict = torch.load(path, weights_only=False, map_location=map_location)
        # 僅加載 policy 部分，推論不需要 optimizer
        resumed_training = self.alg.policy.load_state_dict(loaded_dict["model_state_dict"])
        
        # 加載 RND (若有)
        if hasattr(self.alg, "rnd") and self.alg.rnd:
            self.alg.rnd.load_state_dict(loaded_dict["rnd_state_dict"])
            
        return loaded_dict["infos"]

    def get_inference_policy(self, device: str | None = None) -> callable:
        """獲取推論函數"""
        self.eval_mode()  # 強制切換至 eval 模式
        if device is not None:
            self.alg.policy.to(device)
        return self.alg.policy.act_inference

    def eval_mode(self) -> None:
        """切換為評估模式 (關閉 Dropout 等)"""
        self.alg.policy.eval()
        if hasattr(self.alg, "rnd") and self.alg.rnd:
            self.alg.rnd.eval()

    def _construct_algorithm(self, obs: TensorDict) -> PPO:
        """構建演算法實例 (內部包含 ActorCritic 網路初始化)"""
        # 注意：此處需保留私有函式 resolve_..._config 才能運作
        self.alg_cfg = resolve_rnd_config(self.alg_cfg, obs, self.cfg["obs_groups"], self.env)
        self.alg_cfg = resolve_symmetry_config(self.alg_cfg, self.env)

        # 初始化 Policy 類別 (如 ActorCritic)
        actor_critic_class = eval(self.policy_cfg.pop("class_name"))
        actor_critic = actor_critic_class(
            obs, self.cfg["obs_groups"], self.env.num_actions, **self.policy_cfg
        ).to(self.device)

        # 初始化 PPO 演算法實例
        alg_class = eval(self.alg_cfg.pop("class_name"))
        alg: PPO = alg_class(actor_critic, device=self.device, **self.alg_cfg, multi_gpu_cfg=None)

        return alg

runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)