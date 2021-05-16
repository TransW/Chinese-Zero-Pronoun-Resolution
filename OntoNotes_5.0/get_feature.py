


def get_head_verb(index, wl):
    father = wl[index].parent
    while father:
        leafs = father.get_leaf()
        for ln in leafs:
            if ln.tag.startswith("V"):
                return ln
        father = father.parent

    return None



def get_fl(zp, candidate, wl_zp, wl_candi, wd):
    ifl = []

    (zp_sentence_index, zp_begin_index, zp_end_index) = zp
    (candi_sentence_index, candi_index_begin, candi_index_end) = candidate

    sentence_dis = zp_sentence_index - candi_sentence_index

    # sentence distance
    tmp_ones = [0] * 3
    tmp_ones[sentence_dis] = 1
    ifl += tmp_ones

    cloNP = 0
    if sentence_dis == 0:
        if candi_index_end <= zp_begin_index:
            cloNP = 1
        for i in range(candi_index_end + 1, zp_begin_index):
            node = wl_zp[i]
            while True:
                if node.tag.startswith("NP"):
                    cloNP = 0
                    break
                node = node.parent
                if not node:
                    break
            if cloNP == 0:
                break

    tmp_ones = [0] * 2
    tmp_ones[cloNP] = 1
    ifl += tmp_ones

    first_zp = 1
    for i in range(zp_begin_index):
        if wl_zp[i].word == "*pro*":
            first_zp = 0
            break
    tmp_ones = [0] * 2
    tmp_ones[first_zp] = 1
    ifl += tmp_ones

    last_zp = 1
    for i in range(zp_end_index + 1, len(wl_zp)):
        if wl_zp[i].word == "*pro*":
            last_zp = 0
            break
    tmp_ones = [0] * 2
    tmp_ones[last_zp] = 1
    ifl += tmp_ones

    zp_node = wl_zp[zp_begin_index]
    NP_node = None
    father = zp_node.parent
    while father:
        if father.tag.startswith("NP"):
            NP_node = father
            break
        father = father.parent
    z_NP = 0
    if NP_node:
        z_NP = 1
    tmp_ones = [0] * 2
    tmp_ones[z_NP] = 1
    ifl += tmp_ones

    z_NinI = 0
    if NP_node:
        father = zp_node.parent
        while father:
            if father.tag.startswith("IP"):
                if father.has_child(NP_node):
                    z_NinI = 1
                break
            father = father.parent

    tmp_ones = [0] * 2
    tmp_ones[z_NinI] = 1
    ifl += tmp_ones

    VP_node = None
    zVP = 0
    father = zp_node.parent
    while father:
        if father.tag.startswith("VP"):
            VP_node = father
            zVP = 1
            break
        father = father.parent
    tmp_ones = [0] * 2
    tmp_ones[zVP] = 1
    ifl += tmp_ones

    z_VinI = 0
    if VP_node:
        father = zp_node.parent
        while father:
            if father.tag.startswith("IP"):
                if father.has_child(VP_node):
                    z_VinI = 1
                break
            father = father.parent
    tmp_ones = [0] * 2
    tmp_ones[z_VinI] = 1
    ifl += tmp_ones

    CP_node = None
    zCP = 0
    father = zp_node.parent
    while father:
        if father.tag.startswith("CP"):
            CP_node = father
            zCP = 1
            break
        father = father.parent
    tmp_ones = [0] * 2
    tmp_ones[zCP] = 1
    ifl += tmp_ones

    tags = zp_node.parent.tag.split("-")
    zGram = 0
    zHl = 0
    if len(tags) == 2:
        if tags[1] == "SBJ":
            zGram = 1
        if tags[1] == "HLN":
            zHl = 1
    tmp_ones = [0] * 2
    tmp_ones[zGram] = 1
    ifl += tmp_ones
    tmp_ones = [0] * 2
    tmp_ones[zHl] = 1
    ifl += tmp_ones

    zc = 0
    if zCP == 1:
        zc = 1
        father = zp_node.parent
        while father:
            if father.tag.startswith("IP"):
                zc = 2
                break
            if father == CP_node:
                break
            father = father.parent
    else:
        zc = 3
        father = zp_node.parent
        while father:
            if father.tag.startswith("IP"):
                if father.parent:  # 非根节点
                    zc = 4
                    break
            father = father.parent
    tmp_ones = [0] * 5
    tmp_ones[zc] = 1
    ifl += tmp_ones

    candi_node = wl_candi[candi_index_begin]
    NP_node = None
    father = candi_node.parent
    while father:
        if father.tag.startswith("NP"):
            NP_node = father
            break
        father = father.parent
    can_NinI = 0
    if NP_node:
        father = candi_node.parent
        while father:
            if father.tag.startswith("IP"):
                if father.has_child(NP_node):
                    can_NinI = 1
                break
            father = father.parent
    tmp_ones = [0] * 2
    tmp_ones[can_NinI] = 1
    ifl += tmp_ones
    VP_node = None
    canVP = 0
    father = candi_node.parent
    while father:
        if father.tag.startswith("VP"):
            VP_node = father
            canVP = 1
            break
        father = father.parent
    tmp_ones = [0] * 2
    tmp_ones[canVP] = 1
    ifl += tmp_ones
    can_VinI = 0
    if VP_node:
        father = candi_node.parent
        while father:
            if father.tag.startswith("IP"):
                if father.has_child(VP_node):
                    can_VinI = 1
                break
            father = father.parent
    tmp_ones = [0] * 2
    tmp_ones[can_VinI] = 1
    ifl += tmp_ones
    CP_node = None
    canCP = 0
    father = candi_node.parent
    while father:
        if father.tag.startswith("CP"):
            CP_node = father
            canCP = 1
            break
        father = father.parent
    tmp_ones = [0] * 2
    tmp_ones[canCP] = 1
    ifl += tmp_ones
    tags = candi_node.parent.tag.split("-")
    canGram = 0
    canADV = 0
    canTMP = 0
    canPN = 0
    canHl = 0
    if len(tags) == 2:
        if tags[1] == "SBJ":
            canGram = 1
        elif tags[1] == "OBJ":
            canGram = 2
        if tags[1] == "ADV":
            canADV = 1
        if tags[1] == "TMP":
            canTMP = 1
        if tags[1] == "PN":
            canPN = 1
        if tags[1] == "HLN":
            canHl = 1
    tmp_ones = [0] * 3
    tmp_ones[canGram] = 1
    ifl += tmp_ones
    tmp_ones = [0] * 2
    tmp_ones[canADV] = 1
    ifl += tmp_ones
    tmp_ones = [0] * 2
    tmp_ones[canTMP] = 1
    ifl += tmp_ones
    tmp_ones = [0] * 2
    tmp_ones[canPN] = 1
    ifl += tmp_ones
    tmp_ones = [0] * 2
    tmp_ones[canHl] = 1
    ifl += tmp_ones
    canc = 0
    if canCP == 1:
        canc = 1
        father = candi_node.parent
        while father:
            if father.tag.startswith("IP"):
                canc = 2
                break
            if father == CP_node:
                break
            father = father.parent
    else:
        canc = 3
        father = candi_node.parent
        while father:
            if father.tag.startswith("IP"):
                if father.parent:
                    canc = 4
                    break
            father = father.parent
    tmp_ones = [0] * 5
    tmp_ones[canc] = 1
    ifl += tmp_ones
    sibNV = 0
    if not sentence_dis == 0:
        sibNV = 0
    else:
        if abs(zp_begin_index - candi_index_end) == 1:
            sibNV = 1
        else:
            if abs(zp_begin_index - candi_index_begin) == 1:
                sibNV = 1
            else:
                if abs(zp_begin_index - candi_index_begin) == 2:
                    if zp_begin_index < candi_index_begin:
                        if wl_zp[zp_end_index + 1].tag == "PU":
                            sibNV = 1
                elif abs(zp_begin_index - candi_index_end) == 2:
                    if candi_index_end < zp_begin_index:
                        if wl_zp[zp_begin_index - 1].tag == "PU":
                            sibNV = 1
    tmp_ones = [0] * 2
    tmp_ones[sibNV] = 1
    ifl += tmp_ones
    gram_match = 0
    if not canGram == 0:
        if canGram == zGram:
            gram_match = 1
    tmp_ones = [0] * 2
    tmp_ones[gram_match] = 1
    ifl += tmp_ones

    chv = get_head_verb(candi_index_begin, wl_candi)
    zhv = get_head_verb(zp_begin_index, wl_zp)

    ch = wl_candi[candi_index_end]
    hc = "None"
    pc = "None"
    pz = "None"
    if ch:
        hc = ch.word
    if zhv:
        pz = zhv.word
    if chv:
        pc = chv.word
    tags = candi_node.parent.tag.split("-")
    canGram = "None"
    if len(tags) == 2:
        if tags[1] == "SBJ":
            canGram = "SBJ"
        elif tags[1] == "OBJ":
            canGram = "OBJ"
    gc = canGram
    pcc = "None"
    for i in range(len(wl_zp) - 1, zp_end_index, -1):
        if wl_zp[i].tag.find("PU") >= 0:
            pcc = wl_zp[i].word
            break
    pc_pz = 0
    has = wd.get("%s_%s" % (hc, pcc), 0)

    if pc == pz:
        if canGram == "SBJ":
            pc_pz = 1
        elif canGram == "OBJ":
            pc_pz = 1
        else:
            pc_pz = 2
    tmp_ones = [0] * 3
    tmp_ones[pc_pz] = 1
    ifl += tmp_ones
    tmp_ones = [0] * 2
    tmp_ones[has] = 1
    ifl += tmp_ones
    return ifl







if __name__ == '__main__':
    pass