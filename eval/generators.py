"""답변 생성기 모음. 새 모델은 아래 클래스처럼 만들고 make_generator에 이름을 추가하면 된다.
  __init__(이름)   가볍게: 이름만 저장한다. 모델·클라이언트는 여기서 불러오지 않는다(오타는 바로 알려야 하고, 호출이 없으면 불러올 필요가 없다).
  load()           호출이 필요할 때 한 번 실행: 모델이나 API 클라이언트를 준비한다.
  generate(system_prompt, user_prompt) -> {"text": 답변 문자열, "tok_in": 입력 토큰 수, "tok_out": 출력 토큰 수}  (토큰 수를 모르면 0)
  max_workers      동시에 보낼 요청 수: API는 여러 개, GPU에 올린 로컬 모델은 1.
이름(--generator)은 결과 파일 이름에도 쓰이므로 '/' 같은 문자를 쓰지 않는다.
"""


class ClaudeGenerator:
    max_workers = 8

    def __init__(self, model):
        self.model = model

    def load(self):
        import anthropic
        self.client = anthropic.Anthropic()

    def generate(self, system_prompt, user_prompt):
        response = self.client.messages.create(model=self.model, max_tokens=2048, system=system_prompt,
                                               messages=[{"role": "user", "content": user_prompt}])
        text = "".join(block.text for block in response.content if block.type == "text")
        return {"text": text, "tok_in": response.usage.input_tokens, "tok_out": response.usage.output_tokens}


def make_generator(name):
    if name.startswith("claude-"):
        return ClaudeGenerator(name)
    raise ValueError(f"알 수 없는 생성기 '{name}'. eval/generators.py의 make_generator에 추가하세요.")
