from app.models.types import Word


class HypothesisBuffer:

    def __init__(self):
        self.commited_in_buffer = []
        self.buffer = []
        self.new = []
        self.last_commited_time = 0
        self.last_commited_word = None

    def insert(self, new: list[Word], offset):
        # compare self.commited_in_buffer and new. It inserts only the words in new that extend the commited_in_buffer,
        # it means they are roughly behind last_commited_time and new in content the new tail is added to self.new

        new_time_cutoff = self.last_commited_time - 0.1 - offset
        self.new = [word.add_offset(offset) for word in new if word.start > new_time_cutoff]

        if self.new and self.commited_in_buffer and abs(self.new[0].start - self.last_commited_time) < 1:
            # it's going to search for 1, 2, ..., 5 consecutive words (n-grams) that are identical in commited and new.
            # If they are, they're dropped.# 5 is the maximum
            identical_in_commited_will_compared = min(min(len(self.commited_in_buffer), len(self.new)), 5)

            for i in range(1, identical_in_commited_will_compared + 1):
                c = " ".join([self.commited_in_buffer[-j].word for j in range(1, i + 1)][::-1])
                tail = " ".join(self.new[j - 1].word for j in range(1, i + 1))
                if c == tail:
                    self.new = self.new[i:]
                    break

    def flush(self):
        # returns commited chunk = the longest common prefix of 2 last inserts.
        commit = []
        while self.new:
            word = self.new[0]
            if len(self.buffer) != 0 and word.word == self.buffer[0].word:
                commit.append(word)
                self.last_commited_word = word.word
                self.last_commited_time = word.end
                self.buffer.pop(0)
                self.new.pop(0)
            else:
                break
        self.buffer = self.new
        self.new = []
        self.commited_in_buffer.extend(commit)
        return commit

    def pop_commited(self, time):
        while self.commited_in_buffer and self.commited_in_buffer[0].end <= time:
            self.commited_in_buffer.pop(0)

    def complete(self):
        return self.buffer
