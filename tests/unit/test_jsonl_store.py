from agent_rl.data.schemas import EpisodeRecord, RewardRecord, TurnRecord
from agent_rl.utils.jsonl import JsonlEpisodeStore


def test_episode_jsonl_round_trip(tmp_path):
    episode = EpisodeRecord(
        episode_id="episode",
        group_id="group",
        domain="mock",
        task_id="task",
        model="model",
    )
    episode.append_turn(
        TurnRecord(
            turn_index=0,
            observation="request",
            prompt_messages=[{"role": "user", "content": "request"}],
            action="done",
            next_observation="done",
            terminated=True,
        )
    )
    episode.finish(
        reward=RewardRecord(outcome=1.0, total=1.0),
        success=True,
        termination_reason="user_stop",
    )
    store = JsonlEpisodeStore(tmp_path / "episodes.jsonl")
    store.append(episode)
    loaded = tuple(store.iter_episodes())
    assert loaded[0].to_dict() == episode.to_dict()
