# Walk every input row, parse the dialogue + agent turn, and run the rules.

from .checks import ConversationState, find_violations, has_side_effects
from .console import log
from .tool_schema import parse_available_tools
from .transcript import parse_agent_turn, parse_context


class TurnAnalysis:
    __slots__ = ("index", "example_id", "prompt", "transcript", "turn",
                 "tools", "has_side_effects", "violations")

    def __init__(self, index, example_id, prompt, transcript, turn, tools,
                 has_side_effects=False, violations=None):
        self.index = index
        self.example_id = example_id
        self.prompt = prompt
        self.transcript = transcript
        self.turn = turn
        self.tools = tools
        self.has_side_effects = has_side_effects
        self.violations = [] if violations is None else list(violations)


class DatasetAnalysis:
    __slots__ = ("size", "example_ids", "call_turns", "text_turns")

    def __init__(self, size, example_ids, call_turns, text_turns=None):
        self.size = size
        self.example_ids = example_ids
        self.call_turns = call_turns
        self.text_turns = {} if text_turns is None else text_turns

    @property
    def text_turn_indices(self):
        # Rows without an agent tool call, plus rows that failed to parse.
        return set(range(self.size)) - set(self.call_turns)

    @property
    def violation_indices(self):
        hits = set()
        for row_index, record in self.call_turns.items():
            if record.violations:
                hits.add(row_index)
        return hits

    @property
    def ranking_candidates(self):
        # Call turns that no rule flagged -- these go to the ranking stage.
        return [item for item in self.call_turns.values() if not item.violations]


def _parse_row(row_index, example_id, prompt, response):
    try:
        transcript = parse_context(str(prompt), str(example_id))
        turn = parse_agent_turn(str(response))
    except Exception as error:  # noqa: BLE001 - one broken row must not kill the run
        log("не удалось разобрать %s: %r" % (example_id, error))
        return None
    tools = parse_available_tools(transcript.tools_text)
    return transcript, turn, tools


def analyze_examples(frame):
    text_turns = {}
    call_turns = {}

    paired = zip(frame["id"], frame["prompt"], frame["response"])
    for row_index, (example_id, prompt, response) in enumerate(paired):
        parsed = _parse_row(row_index, example_id, prompt, response)
        if parsed is None:
            continue
        transcript, turn, tools = parsed
        example_id_s = str(example_id)
        prompt_s = str(prompt)

        if not turn.has_tool_calls:
            text_turns[row_index] = TurnAnalysis(
                index=row_index, example_id=example_id_s, prompt=prompt_s,
                transcript=transcript, turn=turn, tools=tools,
                has_side_effects=False,
            )
            continue

        try:
            violations = find_violations(transcript, turn, tools,
                                         ConversationState(transcript))
        except Exception as error:  # noqa: BLE001
            log("проверки упали на %s: %r" % (example_id, error))
            violations = []

        call_turns[row_index] = TurnAnalysis(
            index=row_index, example_id=example_id_s, prompt=prompt_s,
            transcript=transcript, turn=turn, tools=tools,
            has_side_effects=has_side_effects(turn),
            violations=violations,
        )

    ids = [str(x) for x in frame["id"]]
    return DatasetAnalysis(size=len(frame), example_ids=ids,
                           call_turns=call_turns, text_turns=text_turns)
