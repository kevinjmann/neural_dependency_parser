from enum import Enum
from typing import Tuple
from webbrowser import Opera

UD_ID_IDX = 0
UD_LEMMA_IDX = 2
UD_POS_IDX = 3


class UDToken:
    ID = ''
    FORM = ''
    LEMMA = ''
    UPOS = ''
    XPOS = ''
    FEATS = ''
    HEAD = ''
    DEPREL = ''
    DEPS = ''
    MISC = ''

    def __init__(self, columns):
        try:
            [self.ID,
            self.FORM,
            self.LEMMA,
            self.UPOS,
            self.XPOS,
            self.FEATS,
            self.HEAD,
            self.DEPREL,
            self.DEPS,
            self.MISC] = columns
        except:
            print(columns)
            raise Exception("value error")

    def __eq__(self, other):
        return all(
            [
                self.ID == other.ID,
                self.FORM == other.FORM,
                self.LEMMA == other.LEMMA,
                self.UPOS == other.UPOS,
                self.XPOS == other.XPOS,
                self.FEATS == other.FEATS,
                self.HEAD == other.HEAD,
                self.DEPREL == other.DEPREL,
                self.DEPS == other.DEPS,
                self.MISC == other.MISC
            ])
    def __repr__(self):
        return self.FORM

UD_ROOT = UDToken([
    '0',
    'ROOT',
    'ROOT',
    'ROOT',
    '',
    '',
    '0',
    '',
    '',
    ''
])


class Operations(Enum):
    SHIFT = 1
    LEFT = 2
    RIGHT = 3
    INSERT = 4


class Edge:
    members: Tuple = None
    label: str = ''

    def __init__(self, members, label):
        self.members = members
        self.label = label

    def __repr__(self):
        return f'{self.members[0]} -> {self.members[1]}: {self.label}'


def is_crossed(left: UDToken, right: UDToken) -> bool:
    '''
    Used to check if top of stack and bottom of buffer are in cross a
    branch relationship.
    '''

    # get left and right positions for each dependency pair
    left_l = min(int(left.ID), int(left.HEAD))
    left_r = max(int(left.ID), int(left.HEAD))
    right_l = min(int(right.ID), int(right.HEAD))
    right_r = max(int(right.ID), int(right.HEAD))
    right_l_between_left_members = left_l < right_l and right_l < left_r
    return right_l_between_left_members and left_r < right_r


class Parser:
    stack = []
    buffer = []
    edges = []
    operations_taken = []
    root_elem = 'ROOT'

    def __init__(self, root_elem='ROOT'):
        self.root_elem = root_elem
        self.stack = [root_elem]

    def init_parse(self, tokens):
        self.buffer = tokens

    def do_operation(self, operation: Operations, label: str = ''):
        if operation == Operations.SHIFT:
            self.stack.append(self.buffer.pop(0))
        elif operation == Operations.LEFT:
            self.edges.append(
                Edge((self.buffer[0], self.stack.pop()), label))

        elif operation == Operations.RIGHT:
            self.edges.append(Edge((self.stack[-1], self.buffer[0]), label))
            self.buffer[0] = self.stack.pop()
        elif operation == Operations.INSERT:
            stack_top = self.stack.pop()
            self.buffer.insert(1, stack_top)
        self.operations_taken.append((operation, label))

    def __repr__(self):
        edges_str = '\n'.join([str(edge) for edge in self.edges])
        return f'''
stack: {self.stack}
buffer: {self.buffer}
edges: {edges_str}
        '''

    def do_insert_or_shift(self):
        if is_crossed(self.stack[-1], self.buffer[0]):
            self.do_operation(Operations.INSERT)
        else:
            self.do_operation(Operations.SHIFT)

    def step_through_ud(self, ud_text):
        lines = ud_text.split('\n')
        tokens = []
        for line in lines:
            columns = line.strip().split('\t')
            tokens.append(UDToken(columns))

        self.init_parse(tokens)
        max_step_size = 200
        counter = 0
        while True:
            if self.buffer == [UD_ROOT]:
                # parse found
                return True
            # check for left arc
            if len(self.stack) > 0 and len(self.buffer) > 0:
                if self.stack[-1].HEAD == self.buffer[0].ID:
                    can_draw_arc = True
                    for token in self.stack[:-1] + self.buffer[1:]:
                        if token.HEAD == self.stack[-1].ID:
                            can_draw_arc = False
                    if can_draw_arc:
                        self.do_operation(Operations.LEFT, self.stack[-1].DEPREL)
                    else:
                        self.do_insert_or_shift()
                        # self.do_operation(Operations.SHIFT)
                # check for right arc
                elif self.stack[-1].ID == self.buffer[0].HEAD:
                    can_draw_arc = True
                    for token in self.stack[:-1] + self.buffer[1:]:
                        if token.HEAD == self.buffer[0].ID:
                            can_draw_arc = False
                    if can_draw_arc:
                        self.do_operation(Operations.RIGHT, self.buffer[0].DEPREL)
                    else:
                        self.do_insert_or_shift()
                        # self.do_operation(Operations.SHIFT)
                else:
                    self.do_insert_or_shift()
                    # self.do_operation(Operations.SHIFT)
            else:
                if len(self.buffer) > 0:
                    self.do_operation(Operations.SHIFT)

            counter += 1
            if counter > max_step_size:
                return False

    def clear(self):
        self.buffer = []
        self.stack = [self.root_elem]
        self.operations_taken = []
        self.edges = []

    def interactive(self, conllu_text):
        self.buffer = [UDToken(line.strip().split('\t')) for line in conllu_text.split('\n')]
        print(self)
        while True:
            user_input = input(f'input an action to take\n\t1. shift\n\t2. left\n\t3. right\n\t4. insert\n\t "stop" to stop\n')
            if user_input.lower() == 'stop':
                break
            elif user_input == '1':
                self.do_operation(Operations.SHIFT)
            elif user_input == '2':
                self.do_operation(Operations.LEFT)
            elif user_input == '3':
                self.do_operation(Operations.RIGHT)
            elif user_input == '4':
                self.do_operation(Operations.INSERT)
            else:
                print('non recognized input, try again')
            print(self)


def main():
    p = Parser(UD_ROOT)
    print(p)
#     ud_text = '''1	Verlede	verlede	ADJ	ASA	AdjType=Attr|Case=Nom|Degree=Pos	2	amod	_	_
# 2	jaar	jaar	NOUN	NSE	Number=Sing	11	nmod	_	_
# 3	het	het	AUX	VUOT	Tense=Pres|VerbForm=Fin,Inf|VerbType=Aux	11	aux	_	_
# 4	ons	ons	PRON	PEMP	Case=Acc,Nom|Number=Plur|Person=1|PronType=Prs	11	nsubj	_	_
# 5	ons	ons	PRON	PEMB	Number=Plur|Person=1|Poss=Yes|PronType=Prs	7	det	_	_
# 6	eerste	eerste	ADJ	TRAB	AdjType=Attr|Case=Nom|Degree=Pos	7	amod	_	_
# 7	resessie	resessie	NOUN	NSE	Number=Sing	11	obj	_	_
# 8	in	in	ADP	SVS	AdpType=Prep	10	case	_	_
# 9	17	17	SYM	RS	_	10	dep	_	_
# 10	jaar	jaar	NOUN	NSE	Number=Sing	11	obl	_	_
# 11	ervaar	ervaar	VERB	VTHOG	Subcat=Tran|Tense=Pres|VerbForm=Fin,Inf	0	root	_	SpaceAfter=No
# 12	.	.	PUNCT	ZE	_	11	punct	_	_'''
#     ud_text = '''1	एह	एह	DET	DM_DMD	NumType=Card	2	det	_	_
# 2	आयोजन	आयोजन	NOUN	N_NN	Case=Acc|Gender=Masc|Number=Sing|Person=3	6	nmod	_	_
# 3	में	में	ADP	PSP	AdpType=Post	2	case	_	_
# 4	विश्व	विश्व	PROPN	N_NNP	Case=Nom|Gender=Masc|Number=Sing|Person=3	6	compound	_	_
# 5	भोजपुरी	भोजपुरी	PROPN	N_NNP	Case=Nom|Gender=Fem|Number=Sing|Person=3	6	compound	_	SpacesAfter=
# 6	सम्मेलन	सम्मेलन	PROPN	N_NNP	Case=Acc|Gender=Masc|Number=Sing|Person=3	26	nmod	_	_
# 7	,	COMMA	PUNCT	RD_PUNC	_	10	punct	_	_
# 8	पूर्वांचल	पूर्वांचल	PROPN	N_NNP	Case=Nom|Gender=Masc|Number=Sing|Person=3	10	compound	_	_
# 9	एकता	एकता	NOUN	N_NN	Case=Nom|Gender=Masc|Number=Sing|Person=3	10	compound	_	_
# 10	मंच	मंच	NOUN	N_NN	Case=Acc|Gender=Masc|Number=Sing|Person=3	6	conj	_	_
# 11	,	COMMA	PUNCT	RD_PUNC	_	15	punct	_	_
# 12	वीर	वीर	PROPN	N_NNP	Case=Nom|Gender=Masc|Number=Sing|Person=3	15	compound	_	_
# 13	कुँवर	कुँवर	PROPN	N_NNP	Case=Nom|Gender=Masc|Number=Sing|Person=3	15	compound	_	_
# 14	सिंह	सिंह	PROPN	N_NNP	Case=Nom|Gender=Masc|Number=Sing|Person=3	15	compound	_	_
# 15	फाउन्डेशन	फाउन्डेशन	PROPN	N_NNP	Case=Acc|Gender=Masc|Number=Sing|Person=3	6	conj	_	_
# 16	,	COMMA	PUNCT	RD_PUNC	_	19	punct	_	_
# 17	पूर्वांचल	पूर्वांचल	PROPN	N_NNP	Case=Nom|Gender=Masc|Number=Sing|Person=3	19	compound	_	_
# 18	भोजपुरी	भोजपुरी	PROPN	N_NNP	Case=Nom|Gender=Fem|Number=Sing|Person=3	19	compound	_	SpacesAfter=
# 19	महासभा	महासभा	PROPN	N_NNP	Case=Acc|Gender=Fem|Number=Sing|Person=3	6	conj	_	_
# 20	,	COMMA	PUNCT	RD_PUNC	_	22	punct	_	_
# 21	अउर	अउर	CCONJ	CC_CCD	Case=Nom|Gender=Masc|Number=Sing|Person=3	22	nmod	_	_
# 22	हर्फ	हर्फ	NOUN	N_NN	Case=Nom|Gender=Masc|Number=Sing|Person=3	6	conj	_	_
# 23	-	-	PUNCT	RD_SYM	_	22	punct	_	_
# 24	मीडिया	मीडिया	NOUN	N_NN	Case=Acc|Gender=Masc|Number=Sing|Person=3	26	nmod	_	_
# 25	के	का	ADP	PSP	AdpType=Post|Case=Acc|Gender=Masc|Number=Sing	24	case	_	_
# 26	सहभागिता	सहभागिता	NOUN	N_NN	Case=Nom|Gender=Fem|Number=Sing|Person=3	0	root	_	_
# 27	बा	बा	AUX	V_VM	Case=Nom|Gender=Fem|Number=Sing|Person=3	26	cop	_	_
# 28	।	।	PUNCT	RD_PUNC	_	26	punct	_	_'''
    ud_text ='1\tHvor\thvor\tADV\t_\t_\t2\tadvmod\t_\t_\n2\tkommer\tkomme\tVERB\t_\tMood=Ind|Tense=Pres|VerbForm=Fin|Voice=Act\t0\troot\t_\t_\n3\tjulemanden\tjulemand\tNOUN\t_\tDefinite=Def|Gender=Com|Number=Sing\t2\tnsubj\t_\t_\n4\tfra\tfra\tADP\t_\tAdpType=Prep\t1\tcase\t_\tSpaceAfter=No\n5\t?\t?\tPUNCT\t_\t_\t2\tpunct\t_\t_'
    is_parsed = p.step_through_ud(ud_text)
    print(p)
    print(f'operations taken: {p.operations_taken}')

    # sentence = input("input a sentence:")
    # tokens = sentence.split()
    # p = Parser()
    # p.init_parse(tokens)
    # while True:
    #     print(p)
    #     next_action_text = input("what should the parser do next: 'shift', 'left', 'right'? ")
    #     if next_action_text == 'shift'  or next_action_text == 's':
    #         p.do_operation(Operations.SHIFT)
    #     elif next_action_text == 'left' or next_action_text == 'l':
    #         p.do_operation(Operations.LEFT)
    #     elif next_action_text == 'right' or next_action_text == 'r':
    #         p.do_operation(Operations.RIGHT)


if __name__ == '__main__':
    main()
