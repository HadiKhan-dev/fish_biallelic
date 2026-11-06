"""Declare observed join argument types without compilation or filesystem effects.

Signatures are trusted Numba descriptors captured from bounded native fixtures.
Both conventions and one/eight-thread execution contribute to the inventory.

Owns: lazy Numba overload declarations for AS4/AS5 kernels.
Must not: read files, environment, live memory or clocks, or emit logs/warnings.
Method: WP_AS4 §6; C2.
Identity: Declarations describe observed supported calls and do not select scientific routes.
"""

from haplotype_reconstruction.core.kernels import decode_signature

SIGNATURES: dict[str, tuple[dict[str, str], ...]] = {
    "haplotype_reconstruction.science.assembly.join.certify._advance_block": (
        {
            "text": (
                "(Array(float64, 1, 'C', False, aligned=True), int64, int64, int64, int64, int64, Literal"
                "[bool](False), Array(int64, 2, 'C', False, aligned=True), int64, int64, Literal[int](-1)"
                ", Array(float64, 1, 'C', False, aligned=True), bool, int64, float64, Array(float64, 1, '"
                "C', False, aligned=True), int64, Array(float64, 2, 'C', False, aligned=True))"
            ),
            "pickle": (
                "fCQCG0{{R30000ej2UiqZDL_AV{dY0E_8WtWpgfJVsmtIVPkZZj1ymUd2nT4a%E$0ZgX^Ubz^jIa+H&lj0a<HaCv"
                "fNXOxT%Uvgz*Z*FsRa&=>LZ*r8Al#Cf}b!}o{E@N+UWiE7iaAk8YZg6>Yd2nTOl#B&Ia&lpLl#`T<2x4_<Y;<XEb"
                "Ciq*Z(?d?V|0|0lun0~eUvDS1!Q!2aAlNe0%!${7;bfKVqq?0Z*pZWba`-Pb1rjZVQgV?bCiq)Mr?0kbd-~nXbVn"
                "<lzo&aj0A3BZDo{<2WD(<VRSY$l#B>sX>@mKWOQhhOF)bTUt@1%Wt2-Cb%&Hvl#B#!WNB@bO96}qY+-qCb##=B0Y"
                "j8%6pR&Na&lpLC}wPLVRSY$EFdvtEFeQElxP=Cf&q1hlv0#v0%!$j4~z#%Zggd5Wpb30lxPc1hm?JkC}<Rn1!-<{"
                "HZ+uI7E3^k26Jg=Ze?VYh-eo}8g+-1Qj};XXeVeVXeVd_Xa#5wj1EF?Z){~@ZcJ%(WpZI`l#`Tb3r>fWeUvDS4qt"
                "3(bY*g3Y+rU^Y;|RniHrnaYh`(qiHrnpVQpoUj1)|1bY*g3Y+GV)Z){yCMqzAoWhs=51z%%tWM!01y#b69Uu<b~W"
                "pZI`UvznJWnW`qV`ycRXaZ;jXb+4BLT_(uWnpfVlay!+PKT6zlqhHvj09qDZ)}uk7fTX#hm=y3b%&HvlxPBI1!xF"
                "p3r>fWeUvC@4rnK68cPCb8)zJ86pRyLa&lpLC~0nVHZ&|CGGr_uLn)MK7fpx(b%&HvlxQbtCujm_1!xb94oPlwWo"
                "Kn_Olfpwa$#(glay!+PKT6zlqif2Uu<b~WpZI`Uv^<^b!C)F|NsC0j09h6WqFiJ|NsC0j1ymMX>?_BVQgP?d2nT4"
                "V_{=xWt3<qj0A3BZDo{<5KL)wWpZI`TWM}|T_`OvDU^%|Vrg`DX=HS0luJO226Jg=Ze?VYh>QhaV{c?-lugqCb%&"
                "HvlxQDlIA|wm8E7A9Cujm_1!xFp3r>fWeUvC@4rm!@8cPCb8)zJ86pR&Na&lpLC}wPLVRSY$EFdyuEFeQElxP=Ce"
                "F1fclv0#*lr8"
            ),
        },
        {
            "text": (
                "(Array(float64, 1, 'C', False, aligned=True), int64, int64, int64, int64, int64, Literal"
                "[bool](True), Array(int64, 2, 'C', False, aligned=True), int64, int64, int64, Array(floa"
                "t64, 1, 'A', False, aligned=True), bool, int64, float64, Array(float64, 1, 'C', False, a"
                "ligned=True), int64, Array(float64, 2, 'C', False, aligned=True))"
            ),
            "pickle": (
                "fCQE90ssI20000ej2UiqZDL_AV{dY0E_8WtWpgfJVsmtIVPkZZj1ymUd2nT4a%E$0ZgX^Ubz^jIa+H&lj0a<HaCv"
                "fNXOxT%Uvgz*Z*FsRa&=>LZ*r8Al#Cf}b!}o{E@N+UWiE7iaAk8YZg6>Yd2nTOl#B&Ia&lpLl#`T<2x4_<Y;<XEb"
                "Ciq*Z(?d?V|0|0lun0~eUvDS1!Q!2aAlNe0%!${7;bfKVqq?0Z*pZWba`-Pb1rjZVQgV?bCiq)Mr?0kbd-~nXbVn"
                "<lzo&aj0A3BZDo{<2WD(<VRSY$l#B>sX>@mKWOQhhOF)bTUt@1%Wt2-Cb%&Hvl#B#!WNB@bO96}qY+-qCb##=B0Y"
                "j8%6pR&Na&lpLC}wPLVRSY$EFdvtEFeQElxP=Cf&q1hlv0#v0%!$j4~z#%Zggd5Wpb30lxPc1hm?JkC}<Rn1!-<{"
                "HZ+uI7E3^k26Jg=Ze?VYh-eo}8g+-1Qj};XXeVeVXeVd_Xa#5wj1EF?Z){~@ZcJ%(WpZI`l#`Tb3r>fWeUvDS4qt"
                "3(bY*g3Y+rU^Y;|Rnh>QeZYh`(qh>QeoVQpoUj1x?0bY*g3Y+GV)Z){yCRC0A?DU^%_Ut@1%Wt2_60gMx0Y-x05a"
                "$#&=ba`-PUt?ioXl0aW0%!$j4~z#wZ*Od6VQ!R@lxPc1hm?JkC}<Rn1Y&P*Y?NphOA>X5lv0#+hm=y3XaZ;jXb5N"
                "vPKT6zlqhHpXeVeIO9E&cXdGx1j1ysUa$$KWX>N2jG%O%8WGo;<DU@gzO^5+?hm=y3XeVeVXeVd_Xa#5pXbVn<lz"
                "o&aXbxx@Xc|iaXd8?HL6m3|j1^&Wa$$KWW^8X^bT%|BATeYtAVDdVXctX|0d<FzQj};oXeVeHXdh@NXaZ;jXb5Nv"
                "PKT6zlqhHpXc=f4O9E&cXdGx1j1^&Wa$$KWW^8X^bT%|BATneuAVVpXXctX=0d<FzQj~O*E&"
            ),
        },
    ),
    "haplotype_reconstruction.science.assembly.join.certify._gather_flat": (
        {
            "text": (
                "(Array(float64, 1, 'C', False, aligned=True), int64, int64, int64, int64, Array(int64, 2"
                ", 'C', False, aligned=True), int64, Array(float64, 2, 'C', False, aligned=True))"
            ),
            "pickle": (
                "fCQEF0RR910000ej2UiqZDL_AV{dY0E_8WtWpgfJVsmtIVPkZZj1ymUd2nT4a%E$0ZgX^Ubz^jIa+H&lj0a<HaCv"
                "fNXOxT%Uvgz*Z*FsRa&=>LZ*r8Al#Cf}b!}o{E@N+UWiE7iaAk8YZg6>Yd2nTOl#B&Ia&lpLl#`T<2x4_<Y;<XEb"
                "Ciq*Z(?d?V|0|0lun0~eUvDS1!Q!2aAlNe0%!${7;bfKVqq?0Z*pZWba`-Pb1rjZVQgV?bCiq)Mr?0kbd-~nXbVn"
                "<lzo&aj0A3BZDo{<2WD(<VRSY$l#B>sX>@mKWOQhhOF)bTUt@1%Wt2-Cb%&Hvl#B#!WNB@bO96}qY+-qCb##=B0Y"
                "j8%6pR&Na&lpLC}wPLVRSY$EFdvtEFeQElxP=Cf&q1hlv0#v0%!$j4~z#%Zggd5Wpb30lxPc1hm?JkC}<Rn1!-<{"
                "HZ+uI7E3^k26Jg=Ze?VYh-eo}8g+-1Qj};XXeVeVXaZ;jXb5NvPKT6zlqhHpXeVeIO9E&cXdGx1j1ysUa$$KWX>N"
                "2jG%O%8WGo;<DU@gzO^5+?hm=y3XeVd_Xa#5pXbVn<lzo&aXbxx@Xc|ibXd7r8XcUYUVRCX|c_?OVZ((#cG%O%8W"
                "Go;<DU@gzO??4%hm=y3bd)X"
            ),
        },
    ),
    "haplotype_reconstruction.science.assembly.join.certify._gather_move": (
        {
            "text": (
                "(Array(float64, 1, 'C', False, aligned=True), int64, int64, int64, int64, int64, Literal"
                "[bool](False), Array(int64, 2, 'C', False, aligned=True), int64, int64, Literal[int](-1)"
                ", Array(float64, 2, 'C', False, aligned=True))"
            ),
            "pickle": (
                "fCQC40{{R30000ej2UiqZDL_AV{dY0E_8WtWpgfJVsmtIVPkZZj1ymUd2nT4a%E$0ZgX^Ubz^jIa+H&lj0a<HaCv"
                "fNXOxT%Uvgz*Z*FsRa&=>LZ*r8Al#Cf}b!}o{E@N+UWiE7iaAk8YZg6>Yd2nTOl#B&Ia&lpLl#`T<2x4_<Y;<XEb"
                "Ciq*Z(?d?V|0|0lun0~eUvDS1!Q!2aAlNe0%!${7;bfKVqq?0Z*pZWba`-Pb1rjZVQgV?bCiq)Mr?0kbd-~nXbVn"
                "<lzo&aj0A3BZDo{<2WD(<VRSY$l#B>sX>@mKWOQhhOF)bTUt@1%Wt2-Cb%&Hvl#B#!WNB@bO96}qY+-qCb##=B0Y"
                "j8%6pR&Na&lpLC}wPLVRSY$EFdvtEFeQElxP=Cf&q1hlv0#v0%!$j4~z#%Zggd5Wpb30lxPc1hm?JkC}<Rn1!-<{"
                "HZ+uI7E3^k26Jg=Ze?VYh-eo}8g+-1Qj};XXeVeVXeVd_Xa#5wj1EF?Z){~@ZcJ%(WpZI`l#`Tb3r>fWeUvDS4qt"
                "3(bY*g3Y+rU^Y;|RniHrnaYh`(qiHrnpVQpoUj1)|1bY*g3Y+GV)Z){yCMqzAoWhs=51z%%tWM!01y#b69Uu<b~W"
                "pZI`UvznJWnW`qV`ycRXaZ;jXb+4BLT_(uWnpfVlay!+PKT6zlqhHvj09qDZ)}uk7fTX#hm=y3b%&HvlxPBI1!xF"
                "p3r>fWeUvC@4rnK68cPCb8)zJ86pRyLa&lpLC~0nVHZ&|CGGr_uLn)MK7fpx(b%&HvlxQbtCujm_1!xb94oPlwWo"
                "Kn_Olfpwa$#(glay!+PKT6zlqif2Uu<b~WpZI`Uv^<^b!C)F|NsC0j09h6WqFiJ|NsC0j1ymMX>?_BVQgP?d2nT4"
                "V_{=xWt3<qj0A3BZDo{<5KL)wWpZI`TWM}|T_`OvDU^%|Vrg`DX=HS0luJO226Jg=Ze?VYh>QhaV{c?-lugqCb%&"
                "HvlxPBI1!xFp3r>fWeUvC@4rm!@8cPCb8)zJ86pR&Na&lpLC}wPLVRSY$EFdyuEFeQElxP=CeF1fclv0#*lr8"
            ),
        },
        {
            "text": (
                "(Array(float64, 1, 'C', False, aligned=True), int64, int64, int64, int64, int64, Literal"
                "[bool](True), Array(int64, 2, 'C', False, aligned=True), int64, int64, int64, Array(floa"
                "t64, 2, 'C', False, aligned=True))"
            ),
            "pickle": (
                "fCQDK0ssI20000ej2UiqZDL_AV{dY0E_8WtWpgfJVsmtIVPkZZj1ymUd2nT4a%E$0ZgX^Ubz^jIa+H&lj0a<HaCv"
                "fNXOxT%Uvgz*Z*FsRa&=>LZ*r8Al#Cf}b!}o{E@N+UWiE7iaAk8YZg6>Yd2nTOl#B&Ia&lpLl#`T<2x4_<Y;<XEb"
                "Ciq*Z(?d?V|0|0lun0~eUvDS1!Q!2aAlNe0%!${7;bfKVqq?0Z*pZWba`-Pb1rjZVQgV?bCiq)Mr?0kbd-~nXbVn"
                "<lzo&aj0A3BZDo{<2WD(<VRSY$l#B>sX>@mKWOQhhOF)bTUt@1%Wt2-Cb%&Hvl#B#!WNB@bO96}qY+-qCb##=B0Y"
                "j8%6pR&Na&lpLC}wPLVRSY$EFdvtEFeQElxP=Cf&q1hlv0#v0%!$j4~z#%Zggd5Wpb30lxPc1hm?JkC}<Rn1!-<{"
                "HZ+uI7E3^k26Jg=Ze?VYh-eo}8g+-1Qj};XXeVeVXeVd_Xa#5wj1EF?Z){~@ZcJ%(WpZI`l#`Tb3r>fWeUvDS4qt"
                "3(bY*g3Y+rU^Y;|Rnh>QeZYh`(qh>QeoVQpoUj1x?0bY*g3Y+GV)Z){yCRC0A?DU^%_Ut@1%Wt2_60gMx0Y-x05a"
                "$#&=ba`-PUt?ioXl0aW0%!$j4~z#wZ*Od6VQ!R@lxPc1hm?JkC}<Rn1Y&P*Y?NphOA>X5lv0#+hm=y3XaZ;jXb5N"
                "vPKT6zlqhHpXeVeIO9E&cXdGx1j1ysUa$$KWX>N2jG%O%8WGo;<DU@gzO^5+?hm=y3XeVeVXeVd_Xa#5pXbVn<lz"
                "o&aXbxx@Xc|ibXd7r8XcUYUVRCX|c_?OVZ((#cG%O%8WGo;<DU@gzO??4%hm=y3bd)X"
            ),
        },
        {
            "text": (
                "(Array(float64, 1, 'C', False, aligned=True), int64, int64, int64, int64, int64, bool, A"
                "rray(int64, 2, 'C', False, aligned=True), int64, int64, int64, Array(float64, 2, 'C', Fa"
                "lse, aligned=True))"
            ),
            "pickle": (
                "fCQB$0ssI20000ej2UiqZDL_AV{dY0E_8WtWpgfJVsmtIVPkZZj1ymUd2nT4a%E$0ZgX^Ubz^jIa+H&lj0a<HaCv"
                "fNXOxT%Uvgz*Z*FsRa&=>LZ*r8Al#Cf}b!}o{E@N+UWiE7iaAk8YZg6>Yd2nTOl#B&Ia&lpLl#`T<2x4_<Y;<XEb"
                "Ciq*Z(?d?V|0|0lun0~eUvDS1!Q!2aAlNe0%!${7;bfKVqq?0Z*pZWba`-Pb1rjZVQgV?bCiq)Mr?0kbd-~nXbVn"
                "<lzo&aj0A3BZDo{<2WD(<VRSY$l#B>sX>@mKWOQhhOF)bTUt@1%Wt2-Cb%&Hvl#B#!WNB@bO96}qY+-qCb##=B0Y"
                "j8%6pR&Na&lpLC}wPLVRSY$EFdvtEFeQElxP=Cf&q1hlv0#v0%!$j4~z#%Zggd5Wpb30lxPc1hm?JkC}<Rn1!-<{"
                "HZ+uI7E3^k26Jg=Ze?VYh-eo}8g+-1Qj};XXeVeVXeVd_Xa#5wj0Zw*Z){~@Zj_UhXbVn<lzo&aXcUYDVsCG3lxP"
                "=A5_N}^Qj}-{Xa#5pXbVn<lzo&aXbxy6Xc|ibXd7r8XcUYSVRCX|c_?XabT%|BATneuAVVpXXctY00d<FzQj};XX"
                "eVeVXaZ;jXb5NvPKT6zlqhHpXc=f4O9E&cXdGx1j1^&Wa$$KWW^8X^bT%|BATneuAVVpXXctX=0d<FzQj~O*E&"
            ),
        },
    ),
    "haplotype_reconstruction.science.assembly.join.certify._meet": (
        {
            "text": (
                "(Array(float64, 1, 'C', False, aligned=True), Array(float64, 1, 'C', False, aligned=True), int64)"
            ),
            "pickle": (
                "fCQCu0RR910001t8E$oLVqq?0Z*pZWba`-Pb1q?Gb98cHV|0{^6JK<BaAjX|Wn*t{b98cbV{~tFl#`T<2V-w=d2("
                "fEl#C8va%E$0ZgX^Ubz^jIa+H&lj2UiqZDL_AV{dY0E_8WtWpgfWaCvljaAk9pj0Hh*a$$Ltla!1IVs&Y3bZKsLl"
                "#B*%Vrpe$bd-~nPKT6zlqie^WOR9OWt3<FXa$TIZgp*9VJ>5Da%C=bd2nTOE^}jHY+-V9l#B&NY;R$7l#`Tb3r>f"
                "WeUvDS1a4t%Wt5BuW^8X^bT%}Uj0j?Bba!cFbZC@IK#T=nV{c?-luI0Shm=y3j0A3EX>F8C0gMK0VR>(Lbd-z%Lz"
                "HL~j1^&Wa$$KWW^8X^bT%|BATeYtAVVpXXctX_0d<FzQj};PXaZ;jXb+4BNp5syXJvAflay!+PKT6zlqhHvj0I_K"
                "bT%}UXckLAj0SUQXKrO=l!#~-OB!{Dlv0$3lr8"
            ),
        },
    ),
    "haplotype_reconstruction.science.assembly.join.certify._region_block": (
        {
            "text": (
                "(Array(float64, 2, 'C', False, aligned=True), int64, Literal[int](-1), Literal[bool](Tru"
                "e), Array(float64, 1, 'C', False, aligned=True), int64, float64, float64, Array(float64,"
                " 1, 'C', False, aligned=True))"
            ),
            "pickle": (
                "fCQEP0ssI20000ej2UiqZDL_AV{dY0E_8WtWpgfJVsmtIVPkZZj1ymUd2nT4a%E$0ZgX^Ubz^jIa+H&lj0a<HaCv"
                "fNXOxT%Uvgz*Z*FsRa&=>LZ*r8Al#Cf}b!}o{E@N+UWiE7iaAk8YZg6>Yd2nTOl#B&Ia&lpLl#`T<2x4_<Y;<XEb"
                "Ciq*Z(?d?V|0|0lun0~eUvDS1!Q!2aAlNe0%!${7;bfKVqq?0Z*pZWba`-Pb1rjZVQgV?bCiq)Mr?0kbd-~nXbVn"
                "<lzo&aj0A3BZDo{<2WD(<VRSY$l#B>sX>@mKWOQhhOF)bTUt@1%Wt2-Cb%&Hvl#B#!WNB@bO9G4rY+-qCb##=B0Y"
                "j8%6pR&Na&lpLC}wPLVRSY$EFdyuEFeQElxP=CeF1fclv0#v0%!$j4~z#%Zggd5Wpb30lxPc1hm?JkC}<Rn1!-<{"
                "HZ+uI7E3^k26Jg=Ze?VYh-eo}8g+-1Qj}-{Xa#5wj1Ea|bY*8{a!hG-WpZI`l#`Tb3r>fWeUvDS4qt3(bY*g3Y+r"
                "U^Y;|RnO8@`=|BM7*Yh`(qO8@`=|BMq~Y-x05a$#&=ba`-PUt?ioXl0aWCyWGcVQpoUj1Wv|bY*g3Y+Gq=bX_PdF"
                ")5Uc2x4h;cWGpFXp~Dpj0SUQXKrO=l!%N4Ut@1%Wt2_R0d<FzQj}-{Xa#5wj1EF?Z){~@ZcJ%(WpZI`l#`Tb3r>f"
                "WeUvDS4qt3(bY*g3Y+rU^Y;|Rnh>QeZYh`(qh>QeoVQpoUj1x?0bY*g3Y+GV)Z){yCRC0A?DU^%_Ut@1%Wt2_60g"
                "Mx0Y-x05a$#&=ba`-PUt?ioXl0aW0%!$j4~z#wZ*Od6VQ!R@lxPc1hm?JkC}<Rn1Y&P*Y?NphOA>X5lv0#+hm=y3"
                "XaZ;jXb5NvPKT6zlqhHpXc=f4O95ybXdGx1j1^&Wa$$KWW^8X^bT%|BATeYtAVVpXXctX_0d<FzQj};XXc=f3Xia"
                "pKE&"
            ),
        },
        {
            "text": (
                "(Array(float64, 2, 'C', False, aligned=True), int64, int64, bool, Array(float64, 1, 'C',"
                " False, aligned=True), int64, float64, float64, Array(float64, 1, 'C', False, aligned=Tr"
                "ue))"
            ),
            "pickle": (
                "fCQE10RR910000ej2UiqZDL_AV{dY0E_8WtWpgfJVsmtIVPkZZj1ymUd2nT4a%E$0ZgX^Ubz^jIa+H&lj0a<HaCv"
                "fNXOxT%Uvgz*Z*FsRa&=>LZ*r8Al#Cf}b!}o{E@N+UWiE7iaAk8YZg6>Yd2nTOl#B&Ia&lpLl#`T<2x4_<Y;<XEb"
                "Ciq*Z(?d?V|0|0lun0~eUvDS1!Q!2aAlNe0%!${7;bfKVqq?0Z*pZWba`-Pb1rjZVQgV?bCiq)Mr?0kbd-~nXbVn"
                "<lzo&aj0A3BZDo{<2WD(<VRSY$l#B>sX>@mKWOQhhOF)bTUt@1%Wt2-Cb%&Hvl#B#!WNB@bO9G4rY+-qCb##=B0Y"
                "j8%6pR&Na&lpLC}wPLVRSY$EFdyuEFeQElxP=CeF1fclv0#v0%!$j4~z#%Zggd5Wpb30lxPc1hm?JkC}<Rn1!-<{"
                "HZ+uI7E3^k26Jg=Ze?VYh-eo}8g+-1Qj};XXaZ;jXb+4BLT_(uWnpfVlay!+PKT6zlqhHvj09qDZ)}uk7fTX#hm="
                "y3XaZ;jXb5NvPKT6zlqhHpXc=f4O95ybXdGx1j1^&Wa$$KWW^8X^bT%|BATeYtAVVpXXctX_0d<FzQj};XXc=f3X"
                "ft$_E&"
            ),
        },
    ),
    "haplotype_reconstruction.science.assembly.join.certify._region_retreat": (
        {
            "text": (
                "(Array(float64, 2, 'C', False, aligned=True), int64, int64, bool, Array(float64, 1, 'C',"
                " False, aligned=True), int64, float64, Array(float64, 1, 'C', False, aligned=True), Arra"
                "y(float64, 1, 'C', False, aligned=True))"
            ),
            "pickle": (
                "fCQE10RR910000ej2UiqZDL_AV{dY0E_8WtWpgfJVsmtIVPkZZj1ymUd2nT4a%E$0ZgX^Ubz^jIa+H&lj0a<HaCv"
                "fNXOxT%Uvgz*Z*FsRa&=>LZ*r8Al#Cf}b!}o{E@N+UWiE7iaAk8YZg6>Yd2nTOl#B&Ia&lpLl#`T<2x4_<Y;<XEb"
                "Ciq*Z(?d?V|0|0lun0~eUvDS1!Q!2aAlNe0%!${7;bfKVqq?0Z*pZWba`-Pb1rjZVQgV?bCiq)Mr?0kbd-~nXbVn"
                "<lzo&aj0A3BZDo{<2WD(<VRSY$l#B>sX>@mKWOQhhOF)bTUt@1%Wt2-Cb%&Hvl#B#!WNB@bO9G4rY+-qCb##=B0Y"
                "j8%6pR&Na&lpLC}wPLVRSY$EFdyuEFeQElxP=CeF1fclv0#v0%!$j4~z#%Zggd5Wpb30lxPc1hm?JkC}<Rn1!-<{"
                "HZ+uI7E3^k26Jg=Ze?VYh-eo}8g+-1Qj};XXaZ;jXb+4BLT_(uWnpfVlay!+PKT6zlqhHvj09qDZ)}uk7fTX#hm="
                "y3XaZ;jXb5NvPKT6zlqhHpXc=f4O95ybXdGx1j1^&Wa$$KWW^8X^bT%|BATeYtAVVpXXctX_0d<FzQj};XXc=fTX"
                "ft$_E&"
            ),
        },
    ),
    "haplotype_reconstruction.science.assembly.join.certify._retreat_block": (
        {
            "text": (
                "(Array(float64, 1, 'C', False, aligned=True), int64, int64, int64, int64, int64, Literal"
                "[bool](False), Array(int64, 2, 'C', False, aligned=True), int64, int64, Literal[int](-1)"
                ", Array(float64, 1, 'C', False, aligned=True), bool, int64, float64, Array(float64, 1, '"
                "C', False, aligned=True), int64, Array(float64, 2, 'C', False, aligned=True), Array(floa"
                "t64, 1, 'C', False, aligned=True))"
            ),
            "pickle": (
                "fCQCI0{{R30000ej2UiqZDL_AV{dY0E_8WtWpgfJVsmtIVPkZZj1ymUd2nT4a%E$0ZgX^Ubz^jIa+H&lj0a<HaCv"
                "fNXOxT%Uvgz*Z*FsRa&=>LZ*r8Al#Cf}b!}o{E@N+UWiE7iaAk8YZg6>Yd2nTOl#B&Ia&lpLl#`T<2x4_<Y;<XEb"
                "Ciq*Z(?d?V|0|0lun0~eUvDS1!Q!2aAlNe0%!${7;bfKVqq?0Z*pZWba`-Pb1rjZVQgV?bCiq)Mr?0kbd-~nXbVn"
                "<lzo&aj0A3BZDo{<2WD(<VRSY$l#B>sX>@mKWOQhhOF)bTUt@1%Wt2-Cb%&Hvl#B#!WNB@bO96}qY+-qCb##=B0Y"
                "j8%6pR&Na&lpLC}wPLVRSY$EFdvtEFeQElxP=Cf&q1hlv0#v0%!$j4~z#%Zggd5Wpb30lxPc1hm?JkC}<Rn1!-<{"
                "HZ+uI7E3^k26Jg=Ze?VYh-eo}8g+-1Qj};XXeVeVXeVd_Xa#5wj1EF?Z){~@ZcJ%(WpZI`l#`Tb3r>fWeUvDS4qt"
                "3(bY*g3Y+rU^Y;|RniHrnaYh`(qiHrnpVQpoUj1)|1bY*g3Y+GV)Z){yCMqzAoWhs=51z%%tWM!01y#b69Uu<b~W"
                "pZI`UvznJWnW`qV`ycRXaZ;jXb+4BLT_(uWnpfVlay!+PKT6zlqhHvj09qDZ)}uk7fTX#hm=y3b%&HvlxPBI1!xF"
                "p3r>fWeUvC@4rnK68cPCb8)zJ86pRyLa&lpLC~0nVHZ&|CGGr_uLn)MK7fpx(b%&HvlxQbtCujm_1!xb94oPlwWo"
                "Kn_Olfpwa$#(glay!+PKT6zlqif2Uu<b~WpZI`Uv^<^b!C)F|NsC0j09h6WqFiJ|NsC0j1ymMX>?_BVQgP?d2nT4"
                "V_{=xWt3<qj0A3BZDo{<5KL)wWpZI`TWM}|T_`OvDU^%|Vrg`DX=HS0luJO226Jg=Ze?VYh>QhaV{c?-lugqCb%&"
                "HvlxQDlIA|wm8E7A9Cujm_1!xFp3r>fWeUvC@4rm!@8cPCb8)zJ86pR&Na&lpLC}wPLVRSY$EFdyuEFeQElxP=Ce"
                "F1fclv0#vA9R#1"
            ),
        },
        {
            "text": (
                "(Array(float64, 1, 'C', False, aligned=True), int64, int64, int64, int64, int64, Literal"
                "[bool](True), Array(int64, 2, 'C', False, aligned=True), int64, int64, int64, Array(floa"
                "t64, 1, 'A', False, aligned=True), bool, int64, float64, Array(float64, 1, 'C', False, a"
                "ligned=True), int64, Array(float64, 2, 'C', False, aligned=True), Array(float64, 1, 'C',"
                " False, aligned=True))"
            ),
            "pickle": (
                "fCQEB0ssI20000ej2UiqZDL_AV{dY0E_8WtWpgfJVsmtIVPkZZj1ymUd2nT4a%E$0ZgX^Ubz^jIa+H&lj0a<HaCv"
                "fNXOxT%Uvgz*Z*FsRa&=>LZ*r8Al#Cf}b!}o{E@N+UWiE7iaAk8YZg6>Yd2nTOl#B&Ia&lpLl#`T<2x4_<Y;<XEb"
                "Ciq*Z(?d?V|0|0lun0~eUvDS1!Q!2aAlNe0%!${7;bfKVqq?0Z*pZWba`-Pb1rjZVQgV?bCiq)Mr?0kbd-~nXbVn"
                "<lzo&aj0A3BZDo{<2WD(<VRSY$l#B>sX>@mKWOQhhOF)bTUt@1%Wt2-Cb%&Hvl#B#!WNB@bO96}qY+-qCb##=B0Y"
                "j8%6pR&Na&lpLC}wPLVRSY$EFdvtEFeQElxP=Cf&q1hlv0#v0%!$j4~z#%Zggd5Wpb30lxPc1hm?JkC}<Rn1!-<{"
                "HZ+uI7E3^k26Jg=Ze?VYh-eo}8g+-1Qj};XXeVeVXeVd_Xa#5wj1EF?Z){~@ZcJ%(WpZI`l#`Tb3r>fWeUvDS4qt"
                "3(bY*g3Y+rU^Y;|Rnh>QeZYh`(qh>QeoVQpoUj1x?0bY*g3Y+GV)Z){yCRC0A?DU^%_Ut@1%Wt2_60gMx0Y-x05a"
                "$#&=ba`-PUt?ioXl0aW0%!$j4~z#wZ*Od6VQ!R@lxPc1hm?JkC}<Rn1Y&P*Y?NphOA>X5lv0#+hm=y3XaZ;jXb5N"
                "vPKT6zlqhHpXeVeIO9E&cXdGx1j1ysUa$$KWX>N2jG%O%8WGo;<DU@gzO^5+?hm=y3XeVeVXeVd_Xa#5pXbVn<lz"
                "o&aXbxx@Xc|iaXd8?HL6m3|j1^&Wa$$KWW^8X^bT%|BATeYtAVDdVXctX|0d<FzQj};oXeVeHXdh@NXaZ;jXb5Nv"
                "PKT6zlqhHpXc=f4O9E&cXdGx1j1^&Wa$$KWW^8X^bT%|BATneuAVVpXXctX=0d<FzQj};Pbd)X"
            ),
        },
    ),
    "haplotype_reconstruction.science.assembly.join.certify._two_region": (
        {
            "text": (
                "(Array(float64, 1, 'C', False, aligned=True), Array(int64, 1, 'C', False, aligned=True),"
                " Array(int64, 1, 'C', False, aligned=True), Array(int64, 1, 'C', False, aligned=True), A"
                "rray(int64, 2, 'C', False, aligned=True), float64, int64, Array(float64, 1, 'C', False, "
                "aligned=True), bool, int64, int64, Array(int64, 2, 'C', False, aligned=True), Literal[in"
                "t](0), int64, Array(float64, 1, 'C', False, aligned=True), Array(float64, 2, 'C', False,"
                " aligned=True))"
            ),
            "pickle": (
                "fCQEM0ssI20000ej2UiqZDL_AV{dY0E_8WtWpgfJVsmtIVPkZZj1ymUd2nT4a%E$0ZgX^Ubz^jIa+H&lj0a<HaCv"
                "fNXOxT%Uvgz*Z*FsRa&=>LZ*r8Al#Cf}b!}o{E@N+UWiE7iaAk8YZg6>Yd2nTOl#B&Ia&lpLl#`T<2x4_<Y;<XEb"
                "Ciq*Z(?d?V|0|0lun0~eUvDS1!Q!2aAlNe0%!${7;bfKVqq?0Z*pZWba`-Pb1rjZVQgV?bCiq)Mr?0kbd-~nXbVn"
                "<lzo&aj0A3BZDo{<2WD(<VRSY$l#B>sX>@mKWOQhhOF)bTUt@1%Wt2-Cb%&Hvl#B#!WNB@bO96}qY+-qCb##=B0Y"
                "j8%6pR&Na&lpLC}wPLVRSY$EFdvtEFeQElxP=Cf&q1hlv0#v0%!$j2xtpVhm?JkC}<970%!$j4~z#%Zggd5Wpb30"
                "lxPc1hm?JkC}<Rn1!-<{HZ+uI7E3^k26Jg=Ze?VYh-eo}8g+-1Qj};KO95ybXdGx1j1ysUa$$KWX>N2jG%O%7WGo"
                ";<DU@gzO@je-hm=y3Xe?+fXaZ;jXb5NvPKT6zlqhHpXenqKO9E&cXdGx1j1ysUa$$KWX>N2jG%O%8WGo;<DU@gzO"
                "^5+?hm=y3Xc=fJXdh?-Xa#5wj0Zw*Z){~@Zj_UhXbVn<lzo&aXcUYDVsCG3lxP=A5_N}^Qj};ZXenqhXaZ;jXb+4"
                "INp5syXJv9sX>?_BVQiF>lxPc1hm?JkD2xtYY-x05a$#&=c42IFWt2+*j09h6WqFiK0E`n~Y-x05a$#&=ba`-PUt"
                "?ioXl0aWDU1YeVQpoUj1Np{bY*g3Y+Gq=bX_PgDU^%|Vrg`DX=HS0luJO226Jg=Ze?VYh>QhaV{c?-lugV5b%&Hv"
                "lxQhvA7}z-1!xFp3r>fWeUvC@4rm!@8cPCb8)zJ86pR&Na&lpLC}wPLVRSY$EFdyuEFeQElxP=CeF1fclv0#*lr8"
            ),
        },
    ),
    "haplotype_reconstruction.science.assembly.join.certify.birth_bounds": (
        {
            "text": (
                "(Array(float64, 1, 'C', False, aligned=True), Array(int64, 1, 'C', False, aligned=True),"
                " Array(int64, 1, 'C', False, aligned=True), Array(int64, 1, 'C', False, aligned=True), A"
                "rray(int64, 2, 'C', False, aligned=True), float64, Array(float64, 2, 'C', False, aligned"
                "=True), Array(int64, 2, 'C', False, aligned=True), Array(int64, 1, 'C', False, aligned=T"
                "rue), Array(int64, 1, 'C', False, aligned=True), int64, int64, int64)"
            ),
            "pickle": (
                "fCQB|0ssI20000ej2UiqZDL_AV{dY0E_8WtWpgfJVsmtIVPkZZj1ymUd2nT4a%E$0ZgX^Ubz^jIa+H&lj0a<HaCv"
                "fNXOxT%Uvgz*Z*FsRa&=>LZ*r8Al#Cf}b!}o{E@N+UWiE7iaAk8YZg6>Yd2nTOl#B&Ia&lpLl#`T<2x4_<Y;<XEb"
                "Ciq*Z(?d?V|0|0lun0~eUvDS1!Q!2aAlNe0%!${7;bfKVqq?0Z*pZWba`-Pb1rjZVQgV?bCiq)Mr?0kbd-~nXbVn"
                "<lzo&aj0A3BZDo{<2WD(<VRSY$l#B>sX>@mKWOQhhOF)bTUt@1%Wt2-Cb%&Hvl#B#!WNB@bO96}qY+-qCb##=B0Y"
                "j8%6pR&Na&lpLC}wPLVRSY$EFdvtEFeQElxP=Cf&q1hlv0#v0%!$j2xtpVhm?JkC}<970%!$j4~z#%Zggd5Wpb30"
                "lxPc1hm?JkC}<Rn1!-<{HZ+uI7E3^k26Jg=Ze?VYh-eo}8g+-1Qj};KO95ybXdGx1j1ysUa$$KWX>N2jG%O%7WGo"
                ";<DU@gzO@je-hm=y3Xe?+fXaZ;jXb5NvPKT6zlqhHpXenqKO9E&cXdGx1j1ysUa$$KWX>N2jG%O%8WGo;<DU@gzO"
                "^5+?hm=y3Xc=e%Xa#5pXbVn<lzo&aXbxx@Xc|ibXd7r8XcUYUVRCX|c_?OVZ((#cG%O%8WGo;<DU@gzO??4%hm=y"
                "3XfbFkXe?+cXenqZbd)X"
            ),
        },
    ),
    "haplotype_reconstruction.science.assembly.join.certify.forward_covers": (
        {
            "text": (
                "(Array(float64, 1, 'C', False, aligned=True), Array(int64, 1, 'C', False, aligned=True),"
                " Array(int64, 1, 'C', False, aligned=True), Array(int64, 1, 'C', False, aligned=True), A"
                "rray(int64, 2, 'C', False, aligned=True), float64, int64, int64)"
            ),
            "pickle": (
                "fCQED0RR910000ej2UiqZDL_AV{dY0E_8WtWpgfJVsmtIVPkZZj1ymUd2nT4a%E$0ZgX^Ubz^jIa+H&lj0a<HaCv"
                "fNXOxT%Uvgz*Z*FsRa&=>LZ*r8Al#Cf}b!}o{E@N+UWiE7iaAk8YZg6>Yd2nTOl#B&Ia&lpLl#`T<2x4_<Y;<XEb"
                "Ciq*Z(?d?V|0|0lun0~eUvDS1!Q!2aAlNe0%!${7;bfKVqq?0Z*pZWba`-Pb1rjZVQgV?bCiq)Mr?0kbd-~nXbVn"
                "<lzo&aj0A3BZDo{<2WD(<VRSY$l#B>sX>@mKWOQhhOF)bTUt@1%Wt2-Cb%&Hvl#B#!WNB@bO96}qY+-qCb##=B0Y"
                "j8%6pR&Na&lpLC}wPLVRSY$EFdvtEFeQElxP=Cf&q1hlv0#v0%!$j2xtpVhm?JkC}<970%!$j4~z#%Zggd5Wpb30"
                "lxPc1hm?JkC}<Rn1!-<{HZ+uI7E3^k26Jg=Ze?VYh-eo}8g+-1Qj};KO95ybXdGx1j1ysUa$$KWX>N2jG%O%7WGo"
                ";<DU@gzO@je-hm=y3Xe?+fXaZ;jXb5NvPKT6zlqhHpXenqKO9E&cXdGx1j1ysUa$$KWX>N2jG%O%8WGo;<DU@gzO"
                "^5+?hm=y3Xc=fJXeo4*E&"
            ),
        },
    ),
    "haplotype_reconstruction.science.assembly.join.certify.move_bounds": (
        {
            "text": (
                "(Array(float64, 1, 'C', False, aligned=True), Array(int64, 1, 'C', False, aligned=True),"
                " Array(int64, 1, 'C', False, aligned=True), Array(int64, 1, 'C', False, aligned=True), A"
                "rray(int64, 2, 'C', False, aligned=True), float64, int64, Array(float64, 2, 'C', False, "
                "aligned=True), Array(int64, 2, 'C', False, aligned=True), Array(float64, 2, 'C', False, "
                "aligned=True), Array(bool, 3, 'C', False, aligned=True), Array(bool, 3, 'C', False, alig"
                "ned=True), Array(int64, 2, 'C', False, aligned=True), int64, int64, Array(int64, 1, 'C',"
                " False, aligned=True))"
            ),
            "pickle": (
                "fCQDH0ssI20000ej2UiqZDL_AV{dY0E_8WtWpgfJVsmtIVPkZZj1ymUd2nT4a%E$0ZgX^Ubz^jIa+H&lj0a<HaCv"
                "fNXOxT%Uvgz*Z*FsRa&=>LZ*r8Al#Cf}b!}o{E@N+UWiE7iaAk8YZg6>Yd2nTOl#B&Ia&lpLl#`T<2x4_<Y;<XEb"
                "Ciq*Z(?d?V|0|0lun0~eUvDS1!Q!2aAlNe0%!${7;bfKVqq?0Z*pZWba`-Pb1rjZVQgV?bCiq)Mr?0kbd-~nXbVn"
                "<lzo&aj0A3BZDo{<2WD(<VRSY$l#B>sX>@mKWOQhhOF)bTUt@1%Wt2-Cb%&Hvl#B#!WNB@bO96}qY+-qCb##=B0Y"
                "j8%6pR&Na&lpLC}wPLVRSY$EFdvtEFeQElxP=Cf&q1hlv0#v0%!$j2xtpVhm?JkC}<970%!$j4~z#%Zggd5Wpb30"
                "lxPc1hm?JkC}<Rn1!-<{HZ+uI7E3^k26Jg=Ze?VYh-eo}8g+-1Qj};KO95ybXdGx1j1ysUa$$KWX>N2jG%O%7WGo"
                ";<DU@gzO@je-hm=y3Xe?+fXaZ;jXb5NvPKT6zlqhHpXenqKO9E&cXdGx1j1ysUa$$KWX>N2jG%O%8WGo;<DU@gzO"
                "^5+?hm=y3Xc=fJXaZ;jXb5NvPKT6zlqhHpXc=f4O9E&cXdGx1j1^&Wa$$KWW^8X^bT%|BATneuAVVpXXctX=0d<F"
                "zQj};hXf|j9Xa#5pXbVn<lzo&aXbxxsXa#5wj0Zw*Z){~@Zj_UhXbVn<lzo&aXcUYDVsCG3lxP=A5_N}^Qj};KO9"
                "N;dXdGx1j1pmTa$$KWVsCG3EFd#vEFeQElxP=C(E)Xblv0#vLTE8)DQGEZEOeAE"
            ),
        },
    ),
    "haplotype_reconstruction.science.assembly.join.certify.pool_sums": (
        {
            "text": (
                "(Array(float64, 3, 'C', False, aligned=True), Array(float64, 2, 'C', False, aligned=True"
                "), Array(int64, 2, 'C', False, aligned=True))"
            ),
            "pickle": (
                "fCQE40RR910001t8E$oLVqq?0Z*pZWba`-Pb1q?Gb98cHV|0{^6JK<BaAjX|Wn*t{b98cbV{~tFl#`T<2V-w=d2("
                "fEl#C8va%E$0ZgX^Ubz^jIa+H&lj2UiqZDL_AV{dY0E_8WtWpgfWaCvljaAk9pj0Hh*a$$Ltla!1IVs&Y3bZKsLl"
                "#B*%Vrpe$bd-~nPKT6zlqie^WOR9OWt3<FXa$TIZgp*9VJ>5Da%C=bd2nTOE^}jHY+-V9l#B&NY;R$7l#`Tb3r>f"
                "WeUvDS1a4t%Wt5BuW^8X^bT%}Uj0j?Bba!cFbZC@IK#T=nV{c?-luI0Shm=y3j0A3EX>F8C1B?c2VR>(Lbd-z%Lz"
                "HL~j1^&Wa$$KWW^8X^bT%|BATwkvAVVpXXctYT0d<FzQj}-{Xa#5pXbVn<lzo&aXbxx@Xc|ibXd7r8XcUYUVRCX|"
                "c_?OVZ((#cG%O%8WGo;<DU@gzO??4%hm=y3XaZ;jXb5NvPKT6zlqhHpXaZ;jXb+4BNp5syXJvAflay!+PKT6zlqh"
                "Hvj0I_KbT%}UXckLAj0SUQXKrO=l!#~-OB!{Dlv0#v8cPCb8)zJ86pRyLa&lpLC~0nVHZ&|CGGr_uLn)MK7fpx(b"
                "%&Hvl!ufq"
            ),
        },
    ),
    "haplotype_reconstruction.science.assembly.join.certify.region_source": (
        {
            "text": (
                "(Array(int64, 2, 'C', False, aligned=True), Array(bool, 3, 'C', False, aligned=True), Ar"
                "ray(bool, 3, 'C', False, aligned=True), int64, int64, int64, int64)"
            ),
            "pickle": (
                "fCQDU0RR910000ej2UiqZDL_AV{dY0E_8WtWpgfJVsmtIVPkZZj1ymUd2nT4a%E$0ZgX^Ubz^jIa+H&lj0a<HaCv"
                "fNXOxT%Uvgz*Z*FsRa&=>LZ*r8Al#Cf}b!}o{E@N+UWiE7iaAk8YZg6>Yd2nTOl#B&Ia&lpLl#`T<2x4_<Y;<XEb"
                "Ciq*Z(?d?V|0|0lun0~eUvDS1!Q!2aAlNe0%!${7;bfKVqq?0Z*pZWba`-Pb1rjZVQgV?bCiq+Np5syXJvAflay!"
                "+PKT6zlqie@ZeeX@l#B&wZge&@l#B>sX>@mKWOQhhOF)bUb7^O8Wn`3yj0Im~Z)9bZOB!{Dlv0$81a4$$ZInv_j0"
                "S9Bd2e-el#BsGlxP%;6Jc_4VR<NNZge&@EFdyuEFeQElxP@Dhyitnlv0#v0%!$j2xtpVhm?JkC}<970%!$j4~z#w"
                "Z*Od6VQ!R@lxPc1hm?JkC}<Rn1Y&P*Y?NpiOA>X5lv0#v8%qOd9B3VA6pRvKa&lpLC}MAKY%Cx%WGo;<DU@g!P0;"
                "~$hm=y3Xe?+NXc}l5Xc}~sE&"
            ),
        },
        {
            "text": (
                "(Array(int64, 2, 'C', False, aligned=True), Array(bool, 3, 'C', False, aligned=True), Ar"
                "ray(bool, 3, 'C', False, aligned=True), uint64, int64, int64, int64)"
            ),
            "pickle": (
                "fCQD*0RR910000ej2UiqZDL_AV{dY0E_8WtWpgfJVsmtIVPkZZj1ymUd2nT4a%E$0ZgX^Ubz^jIa+H&lj0a<HaCv"
                "fNXOxT%Uvgz*Z*FsRa&=>LZ*r8Al#Cf}b!}o{E@N+UWiE7iaAk8YZg6>Yd2nTOl#B&Ia&lpLl#`T<2x4_<Y;<XEb"
                "Ciq*Z(?d?V|0|0lun0~eUvDS1!Q!2aAlNe0%!${7;bfKVqq?0Z*pZWba`-Pb1rjZVQgV?bCiq+Np5syXJvAflay!"
                "+PKT6zlqie@ZeeX@l#B&wZge&@l#B>sX>@mKWOQhhOF)bUb7^O8Wn`3yj0Im~Z)9bZOB!{Dlv0$81a4$$ZInv_j0"
                "S9Bd2e-el#BsGlxP%;6Jc_4VR<NNZge&@EFdyuEFeQElxP@Dhyitnlv0#v0%!$j2xtpVhm?JkC}<970%!$j4~z#w"
                "Z*Od6VQ!R@lxPc1hm?JkC}<Rn1Y&P*Y?NpiOA>X5lv0#v8%qOd9B3VA6pRvKa&lpLC}MAKY%Cx%WGo;<DU@g!P0;"
                "~$hm=y3Xe?*~Xa#5yXbVn<lzo&aXcUYFb!l#NHZ+uI7E3^A7l~*XOBQv9lv0#v8fY458g!H{"
            ),
        },
    ),
    "haplotype_reconstruction.science.assembly.join.certify.replace_bounds": (
        {
            "text": (
                "(Array(float64, 1, 'C', False, aligned=True), Array(int64, 1, 'C', False, aligned=True),"
                " Array(int64, 1, 'C', False, aligned=True), Array(int64, 1, 'C', False, aligned=True), A"
                "rray(int64, 2, 'C', False, aligned=True), float64, Array(int64, 2, 'C', False, aligned=T"
                "rue), Array(float64, 2, 'C', False, aligned=True), Array(int64, 1, 'C', False, aligned=T"
                "rue), int64, Array(int64, 2, 'C', False, aligned=True), Array(int64, 1, 'C', False, alig"
                "ned=True), Array(int64, 1, 'C', False, aligned=True), Array(int64, 2, 'C', False, aligne"
                "d=True), Array(int64, 1, 'C', False, aligned=True), Array(int64, 1, 'C', False, aligned="
                "True), int64, Array(bool, 1, 'C', False, aligned=True))"
            ),
            "pickle": (
                "fCQDL0ssI20000ej2UiqZDL_AV{dY0E_8WtWpgfJVsmtIVPkZZj1ymUd2nT4a%E$0ZgX^Ubz^jIa+H&lj0a<HaCv"
                "fNXOxT%Uvgz*Z*FsRa&=>LZ*r8Al#Cf}b!}o{E@N+UWiE7iaAk8YZg6>Yd2nTOl#B&Ia&lpLl#`T<2x4_<Y;<XEb"
                "Ciq*Z(?d?V|0|0lun0~eUvDS1!Q!2aAlNe0%!${7;bfKVqq?0Z*pZWba`-Pb1rjZVQgV?bCiq)Mr?0kbd-~nXbVn"
                "<lzo&aj0A3BZDo{<2WD(<VRSY$l#B>sX>@mKWOQhhOF)bTUt@1%Wt2-Cb%&Hvl#B#!WNB@bO96}qY+-qCb##=B0Y"
                "j8%6pR&Na&lpLC}wPLVRSY$EFdvtEFeQElxP=Cf&q1hlv0#v0%!$j2xtpVhm?JkC}<970%!$j4~z#%Zggd5Wpb30"
                "lxPc1hm?JkC}<Rn1!-<{HZ+uI7E3^k26Jg=Ze?VYh-eo}8g+-1Qj};KO95ybXdGx1j1ysUa$$KWX>N2jG%O%7WGo"
                ";<DU@gzO@je-hm=y3Xe?+fXaZ;jXb5NvPKT6zlqhHpXenqKO9E&cXdGx1j1ysUa$$KWX>N2jG%O%8WGo;<DU@gzO"
                "^5+?hm=y3Xc=fRXaZ;jXb5NvPKT6zlqhHpXc=f4O9E&cXdGx1j1^&Wa$$KWW^8X^bT%|BATneuAVVpXXctX=0d<F"
                "zQj};cXenqhXe?+fXfbFkXe?+cXaZ;jXb5NvPKT6zlqhHpXaZ;jXb+4BLT_(uWnpfVlay!+PKT6zlqhHvj09qDZ)"
                "}uk7fTX#hm=y3Xc|iaXd7r8XcUYRVRCX|c_?CUZ)_|eF=Q+tLn)MK7fq=Fb%&HvlysCX"
            ),
        },
    ),
    "haplotype_reconstruction.science.assembly.join.certify.single_exclusions": (
        {
            "text": (
                "(Array(float64, 1, 'C', False, aligned=True), Array(int64, 1, 'C', False, aligned=True),"
                " Array(int64, 1, 'C', False, aligned=True), Array(int64, 1, 'C', False, aligned=True), A"
                "rray(int64, 2, 'C', False, aligned=True), float64, int64, Array(int64, 2, 'C', False, al"
                "igned=True), int64)"
            ),
            "pickle": (
                "fCQEF0RR910000ej2UiqZDL_AV{dY0E_8WtWpgfJVsmtIVPkZZj1ymUd2nT4a%E$0ZgX^Ubz^jIa+H&lj0a<HaCv"
                "fNXOxT%Uvgz*Z*FsRa&=>LZ*r8Al#Cf}b!}o{E@N+UWiE7iaAk8YZg6>Yd2nTOl#B&Ia&lpLl#`T<2x4_<Y;<XEb"
                "Ciq*Z(?d?V|0|0lun0~eUvDS1!Q!2aAlNe0%!${7;bfKVqq?0Z*pZWba`-Pb1rjZVQgV?bCiq)Mr?0kbd-~nXbVn"
                "<lzo&aj0A3BZDo{<2WD(<VRSY$l#B>sX>@mKWOQhhOF)bTUt@1%Wt2-Cb%&Hvl#B#!WNB@bO96}qY+-qCb##=B0Y"
                "j8%6pR&Na&lpLC}wPLVRSY$EFdvtEFeQElxP=Cf&q1hlv0#v0%!$j2xtpVhm?JkC}<970%!$j4~z#%Zggd5Wpb30"
                "lxPc1hm?JkC}<Rn1!-<{HZ+uI7E3^k26Jg=Ze?VYh-eo}8g+-1Qj};KO95ybXdGx1j1ysUa$$KWX>N2jG%O%7WGo"
                ";<DU@gzO@je-hm=y3Xe?+fXaZ;jXb5NvPKT6zlqhHpXenqKO9E&cXdGx1j1ysUa$$KWX>N2jG%O%8WGo;<DU@gzO"
                "^5+?hm=y3Xc=fJXfbFhbd)X"
            ),
        },
    ),
    "haplotype_reconstruction.science.assembly.join.certify.structural_bounds": (
        {
            "text": (
                "(Array(float64, 1, 'C', False, aligned=True), Array(int64, 1, 'C', False, aligned=True),"
                " Array(int64, 1, 'C', False, aligned=True), Array(int64, 1, 'C', False, aligned=True), A"
                "rray(int64, 2, 'C', False, aligned=True), float64, int64, Array(float64, 2, 'C', False, "
                "aligned=True), Array(int64, 2, 'C', False, aligned=True), Array(float64, 2, 'C', False, "
                "aligned=True), Array(bool, 3, 'C', False, aligned=True), Array(bool, 3, 'C', False, alig"
                "ned=True), Array(int64, 1, 'C', False, aligned=True), Array(bool, 1, 'C', False, aligned"
                "=True), Array(int64, 2, 'C', False, aligned=True), Array(int64, 3, 'C', False, aligned=T"
                "rue), Array(int64, 3, 'C', False, aligned=True), int64)"
            ),
            "pickle": (
                "fCQBo0{{R30000ej2UiqZDL_AV{dY0E_8WtWpgfJVsmtIVPkZZj1ymUd2nT4a%E$0ZgX^Ubz^jIa+H&lj0a<HaCv"
                "fNXOxT%Uvgz*Z*FsRa&=>LZ*r8Al#Cf}b!}o{E@N+UWiE7iaAk8YZg6>Yd2nTOl#B&Ia&lpLl#`T<2x4_<Y;<XEb"
                "Ciq*Z(?d?V|0|0lun0~eUvDS1!Q!2aAlNe0%!${7;bfKVqq?0Z*pZWba`-Pb1rjZVQgV?bCiq)Mr?0kbd-~nXbVn"
                "<lzo&aj0A3BZDo{<2WD(<VRSY$l#B>sX>@mKWOQhhOF)bTUt@1%Wt2-Cb%&Hvl#B#!WNB@bO96}qY+-qCb##=B0Y"
                "j8%6pR&Na&lpLC}wPLVRSY$EFdvtEFeQElxP=Cf&q1hlv0#v0%!$j2xtpVhm?JkC}<970%!$j4~z#%Zggd5Wpb30"
                "lxPc1hm?JkC}<Rn1!-<{HZ+uI7E3^k26Jg=Ze?VYh-eo}8g+-1Qj};KO95ybXdGx1j1ysUa$$KWX>N2jG%O%7WGo"
                ";<DU@gzO@je-hm=y3Xe?+fXaZ;jXb5NvPKT6zlqhHpXenqKO9E&cXdGx1j1ysUa$$KWX>N2jG%O%8WGo;<DU@gzO"
                "^5+?hm=y3Xc=fJXaZ;jXb5NvPKT6zlqhHpXc=f4O9E&cXdGx1j1^&Wa$$KWW^8X^bT%|BATneuAVVpXXctX=0d<F"
                "zQj};hXf|j9Xa#5pXbVn<lzo&aXbxxsXa#5wj0Zw*Z){~@Zj_UhXbVn<lzo&aXcUYDVsCG3lxP=A5_N}^Qj};KO9"
                "N;dXdGx1j1pmTa$$KWVsCG3EFd#vEFeQElxP=C(E)Xblv0#vLTD^#0%!$j2xtpVhm?JkC}<97KWG|D0caa&9B34b"
                "5@B+3VR<NGZ*OcYATeYtAVVpXXctYX0d<FzQj};hXaZ;jXb5NvPKT6zlqhHpXenqKO9N;dXdGx1j1ysUa$$KWX>N"
                "2jG%O%9WGo;<DU@gzP1*r<hm=y3XiR7+bd)X"
            ),
        },
    ),
    "haplotype_reconstruction.science.assembly.join.emissions._binned_log_emissions": (
        {
            "text": (
                "(Array(float64, 1, 'C', False, aligned=True), int64, Array(float32, 3, 'C', False, align"
                "ed=True), int64, Array(int8, 2, 'C', False, aligned=True), Array(int32, 2, 'C', False, a"
                "ligned=True), Array(bool, 1, 'C', False, aligned=True), Array(bool, 1, 'C', False, align"
                "ed=True), int64)"
            ),
            "pickle": (
                "fCQBS0{{R30000ej2UiqZDL_AV{dY0E_8WtWpgfJVsmtIVPkZZj1ymUd2nT4a%E$0ZgX^Ubz^jIa+H&lj0a<HaCv"
                "fNXOxT%Uvgz*Z*FsRa&=>LZ*r8Al#Cf}b!}o{E@N+UWiE7iaAk8YZg6>Yd2nTOl#B&Ia&lpLl#`T<2x4_<Y;<XEb"
                "Ciq*Z(?d?V|0|0lun0~eUvDS1!Q!2aAlNe0%!${7;bfKVqq?0Z*pZWba`-Pb1rjZVQgV?bCiq)Mr?0kbd-~nXbVn"
                "<lzo&aj0A3BZDo{<2WD(<VRSY$l#B>sX>@mKWOQhhOF)bTUt@1%Wt2-Cb%&Hvl#B#!WNB@bO96}qY+-qCb##=B0Y"
                "j8%6pR&Na&lpLC}wPLVRSY$EFdvtEFeQElxP=Cf&q1hlv0#v0%!$j4~z#%Zggd5Wpb30lxPc1hm?JkC}<Rn1!-<{"
                "HZ+uI7E3^k26Jg=Ze?VYh-eo}8g+-1Qj}-{Xa#5pXbVn<lzo&aXbxxsXa#5yXbVn<lzo&aXcUYGW^8X^bTcxPXck"
                "K#XctQxb%&HvlxP}D185s)9B34b6=8C6VR<NKY;R$7Gcqh7Gh{3vLn)MK7fl=kb%&HvlxQbt0%!$j2xtpVhm?JkC"
                "}<970%!$jA!rLuhm?JkC}<Rn1Zi$`IFx7>O9*Hsh-eo}7j=h}Qj};KO9E&cXdGx1j1pmTa$$KWX>N2lEFdyuEFeQ"
                "ElxP=C9RqcTlv0#v0%!$j2xtpVhm?JkC}<970%!$jA!rLuhm?JkC}<Rn1!-<{GcuHD7E2&#C5UJjOBr>Clv0#v8c"
                "PCb8)zJ86pRyLa&lpLC~0nVGcqh7GGr_uLn)MK7fl`mb%&HvlxPBI1!xFp3r>fWeUvC@4rl^s1!xb92SRUeY-M3?"
                "l#`Tb3r>fWeUvC@6pRF7Z*OdrXctQob%&HvlxP}D0caa&9B34b5@B+3VR<NGZ*OcYATeYtAVVpXXctW%19gX#Qj}"
                ";>XeV@(E&"
            ),
        },
        {
            "text": (
                "(Array(float64, 1, 'C', False, aligned=True), int64, Array(float32, 3, 'C', True, aligne"
                "d=True), int64, Array(int8, 2, 'C', False, aligned=True), Array(int32, 2, 'C', False, al"
                "igned=True), Array(bool, 1, 'C', False, aligned=True), Array(bool, 1, 'C', False, aligne"
                "d=True), int64)"
            ),
            "pickle": (
                "fCQBm0{{R30000ej2UiqZDL_AV{dY0E_8WtWpgfJVsmtIVPkZZj1ymUd2nT4a%E$0ZgX^Ubz^jIa+H&lj0a<HaCv"
                "fNXOxT%Uvgz*Z*FsRa&=>LZ*r8Al#Cf}b!}o{E@N+UWiE7iaAk8YZg6>Yd2nTOl#B&Ia&lpLl#`T<2x4_<Y;<XEb"
                "Ciq*Z(?d?V|0|0lun0~eUvDS1!Q!2aAlNe0%!${7;bfKVqq?0Z*pZWba`-Pb1rjZVQgV?bCiq)Mr?0kbd-~nXbVn"
                "<lzo&aj0A3BZDo{<2WD(<VRSY$l#B>sX>@mKWOQhhOF)bTUt@1%Wt2-Cb%&Hvl#B#!WNB@bO96}qY+-qCb##=B0Y"
                "j8%6pR&Na&lpLC}wPLVRSY$EFdvtEFeQElxP=Cf&q1hlv0#v0%!$j4~z#%Zggd5Wpb30lxPc1hm?JkC}<Rn1!-<{"
                "HZ+uI7E3^k26Jg=Ze?VYh-eo}8g+-1Qj}-{Xa#5pXbVn<lzo&aj0bIXbYWs_Wt53%4rl^s1!xgy3r>fWeUvC@6pR"
                "OEY;R$7GcuHD7E2&#7fTy;hm=y3Xc|icXd7r8XcUYda%Ev;Z*FXPAYpQHVR<NKY;R$7Gcqh7Gh{3vLn)MK7fl=kb"
                "%&HvlxQbt0%!$j2xtpVhm?JkC}<970%!$jA!rLuhm?JkC}<Rn1Zi$`IFx7>O9*Hsh-eo}7j=h}Qj};KO9E&cXdGx"
                "1j1pmTa$$KWX>N2lEFdyuEFeQElxP=C9RqcTlv0#v0%!$j2xtpVhm?JkC}<970%!$jA!rLuhm?JkC}<Rn1!-<{Gc"
                "uHD7E2&#C5UJjOBr>Clv0#v8cPCb8)zJ86pRyLa&lpLC~0nVGcqh7GGr_uLn)MK7fl`mb%&HvlxPBI1!xFp3r>fW"
                "eUvC@4rl^s1!xb92SRUeY-M3?l#`Tb3r>fWeUvC@6pRF7Z*OdrXctQob%&HvlxP}D0caa&9B34b5@B+3VR<NGZ*O"
                "cYATeYtAVVpXXctW%19gX#Qj};?XeV@(E&"
            ),
        },
        {
            "text": (
                "(Array(float64, 1, 'C', False, aligned=True), int64, Array(float32, 3, 'C', True, aligne"
                "d=True), int64, Array(int8, 2, 'C', True, aligned=True), Array(int32, 2, 'C', False, ali"
                "gned=True), Array(bool, 1, 'C', False, aligned=True), Array(bool, 1, 'C', False, aligned"
                "=True), int64)"
            ),
            "pickle": (
                "fCQBy0{{R30000ej2UiqZDL_AV{dY0E_8WtWpgfJVsmtIVPkZZj1ymUd2nT4a%E$0ZgX^Ubz^jIa+H&lj0a<HaCv"
                "fNXOxT%Uvgz*Z*FsRa&=>LZ*r8Al#Cf}b!}o{E@N+UWiE7iaAk8YZg6>Yd2nTOl#B&Ia&lpLl#`T<2x4_<Y;<XEb"
                "Ciq*Z(?d?V|0|0lun0~eUvDS1!Q!2aAlNe0%!${7;bfKVqq?0Z*pZWba`-Pb1rjZVQgV?bCiq)Mr?0kbd-~nXbVn"
                "<lzo&aj0A3BZDo{<2WD(<VRSY$l#B>sX>@mKWOQhhOF)bTUt@1%Wt2-Cb%&Hvl#B#!WNB@bO96}qY+-qCb##=B0Y"
                "j8%6pR&Na&lpLC}wPLVRSY$EFdvtEFeQElxP=Cf&q1hlv0#v0%!$j4~z#%Zggd5Wpb30lxPc1hm?JkC}<Rn1!-<{"
                "HZ+uI7E3^k26Jg=Ze?VYh-eo}8g+-1Qj}-{Xa#5pXbVn<lzo&aj0bIXbYWs_Wt53%4rl^s1!xgy3r>fWeUvC@6pR"
                "OEY;R$7GcuHD7E2&#7fTy;hm=y3Xc|icXd7r8XcUYda%Ev;Z*FXPAYpQHVR<NKY;R$7Gcqh7Gh{3vLn)MK7fp@<b"
                "%&HvlxQbt0%!$j2xtpVhm?JkC}=8)XbxxsXa#5?XbVn<lzo&aXcUYDX>N2llxP-92xujUXctQtb%&HvlxP}D0%#j"
                "(9B34b8**h~WN&V4c_3kOa$$KWX>N2lEFdyuEFeQElxP=CArEzjlv0#v0%!$j2xtpVhm?JkC}<970%!$jA!rLuhm"
                "?JkC}<Rn1!-<{GcuHD7E2&#C5UJjOBr>Clv0#v8cPCb8)zJ86pRyLa&lpLC~0nVGcqh7GGr_uLn)MK7fqJ|b%&Hv"
                "lxPBI1!xFp3r>fWeUvC@4rl^s1!xb92SRUeY-M3?l#`Tb3r>fWeUvC@6pRF7Z*OdrXctQob%&HvlxP}D0caa&9B3"
                "4b5@B+3VR<NGZ*OcYATeYtAVVpXXctYX0d<FzQj};?XeV@(E&"
            ),
        },
    ),
    "haplotype_reconstruction.science.assembly.join.objective._advance": (
        {
            "text": (
                "(Array(float64, 1, 'C', False, aligned=True), int64, Array(float64, 2, 'C', False, align"
                "ed=True), int64, float64)"
            ),
            "pickle": (
                "fCQDa0RR910000ej2UiqZDL_AV{dY0E_8WtWpgfJVsmtIVPkZZj1ymUd2nT4a%E$0ZgX^Ubz^jIa+H&lj0a<HaCv"
                "fNXOxT%Uvgz*Z*FsRa&=>LZ*r8Al#Cf}b!}o{E@N+UWiE7iaAk8YZg6>Yd2nTOl#B&Ia&lpLl#`T<2x4_<Y;<XEb"
                "Ciq*Z(?d?V|0|0lun0~eUvDS1!Q!2aAlNe0%!${7;bfKVqq?0Z*pZWba`-Pb1rjZVQgV?bCiq)Mr?0kbd-~nXbVn"
                "<lzo&aj0A3BZDo{<2WD(<VRSY$l#B>sX>@mKWOQhhOF)bTUt@1%Wt2-Cb%&Hvl#B#!WNB@bO96}qY+-qCb##=B0Y"
                "j8%6pR&Na&lpLC}wPLVRSY$EFdvtEFeQElxP=Ce*txelv0#v0%!$j4~z#%Zggd5Wpb30lxPc1hm?JkC}<Rn1!-<{"
                "HZ+uI7E3^k26Jg=Ze?VYh-eo}8g+-1Qj}-{Xa#5pXbVn<lzo&aXbxx@Xc|ibXd7r8XcUYUVRCX|c_?OVZ((#cG%O"
                "%8WGo;<DU@gzO_2e0hm=y3XeVeHbd)X"
            ),
        },
    ),
    "haplotype_reconstruction.science.assembly.join.objective._gather_states": (
        {
            "text": (
                "(Array(float64, 1, 'C', False, aligned=True), int64, int64, int64, Array(int64, 2, 'C', "
                "False, aligned=True), Array(int64, 1, 'C', False, aligned=True), Array(int64, 1, 'C', Fa"
                "lse, aligned=True), int64, Array(float64, 2, 'C', False, aligned=True))"
            ),
            "pickle": (
                "fCQB=0ssI20000ej2UiqZDL_AV{dY0E_8WtWpgfJVsmtIVPkZZj1ymUd2nT4a%E$0ZgX^Ubz^jIa+H&lj0a<HaCv"
                "fNXOxT%Uvgz*Z*FsRa&=>LZ*r8Al#Cf}b!}o{E@N+UWiE7iaAk8YZg6>Yd2nTOl#B&Ia&lpLl#`T<2x4_<Y;<XEb"
                "Ciq*Z(?d?V|0|0lun0~eUvDS1!Q!2aAlNe0%!${7;bfKVqq?0Z*pZWba`-Pb1rjZVQgV?bCiq)Mr?0kbd-~nXbVn"
                "<lzo&aj0A3BZDo{<2WD(<VRSY$l#B>sX>@mKWOQhhOF)bTUt@1%Wt2-Cb%&Hvl#B#!WNB@bO96}qY+-qCb##=B0Y"
                "j8%6pR&Na&lpLC}wPLVRSY$EFdvtEFeQElxP=Ce*txelv0#v0%!$j4~z#%Zggd5Wpb30lxPc1hm?JkC}<Rn1!-<{"
                "HZ+uI7E3^k26Jg=Ze?VYh-eo}8g+-1Qj};XXeVd_Xa#5pXbVn<lzo&aXbxy6Xc|ibXd7r8XcUYSVRCX|c_?XabT%"
                "|BATneuAVVpXXctYU0d<FzQj}-{Xa#5pXbVn<lzo&aXbxy6Xc|iaXd7r8XcUYSVRCX|c_?XabT%|BATeYtAVVpXX"
                "ctYE0d<FzQj};hXeVd_Xa#5pXbVn<lzo&aXbxx@Xc|ibXd7r8XcUYUVRCX|c_?OVZ((#cG%O%8WGo;<DU@gzO_2e"
                "0hm=y3bd)X"
            ),
        },
    ),
    "haplotype_reconstruction.science.assembly.join.objective._start": (
        {
            "text": (
                "(Array(float64, 1, 'C', False, aligned=True), int64, Array(float64, 2, 'C', False, align"
                "ed=True), int64)"
            ),
            "pickle": (
                "fCQDY0RR910000ej2UiqZDL_AV{dY0E_8WtWpgfJVsmtIVPkZZj1ymUd2nT4a%E$0ZgX^Ubz^jIa+H&lj0a<HaCv"
                "fNXOxT%Uvgz*Z*FsRa&=>LZ*r8Al#Cf}b!}o{E@N+UWiE7iaAk8YZg6>Yd2nTOl#B&Ia&lpLl#`T<2x4_<Y;<XEb"
                "Ciq*Z(?d?V|0|0lun0~eUvDS1!Q!2aAlNe0%!${7;bfKVqq?0Z*pZWba`-Pb1rjZVQgV?bCiq)Mr?0kbd-~nXbVn"
                "<lzo&aj0A3BZDo{<2WD(<VRSY$l#B>sX>@mKWOQhhOF)bTUt@1%Wt2-Cb%&Hvl#B#!WNB@bO96}qY+-qCb##=B0Y"
                "j8%6pR&Na&lpLC}wPLVRSY$EFdvtEFeQElxP=Ce*txelv0#v0%!$j4~z#%Zggd5Wpb30lxPc1hm?JkC}<Rn1!-<{"
                "HZ+uI7E3^k26Jg=Ze?VYh-eo}8g+-1Qj}-{Xa#5pXbVn<lzo&aXbxx@Xc|ibXd7r8XcUYUVRCX|c_?OVZ((#cG%O"
                "%8WGo;<DU@gzO_2e0hm=y3XeV@(E&"
            ),
        },
    ),
    "haplotype_reconstruction.science.assembly.join.objective.column_sums": (
        {
            "text": ("(Array(float64, 2, 'C', False, aligned=True),)"),
            "pickle": (
                "fCQB`0RR910001t8E$oLVqq?0Z*pZWba`-Pb1q?Gb98cHV|0{^6JK<BaAjX|Wn*t{b98cbV{~tFl#`T<2V-w=d2("
                "fEl#C8va%E$0ZgX^Ubz^jIa+H&lj2UiqZDL_AV{dY0E_8WtWpgfWaCvljaAk9pj0Hh*a$$Ltla!1IVs&Y3bZKsLl"
                "#B*%Vrpe$bd-~nPKT6zlqie^WOR9OWt3<FXa$TIZgp*9VJ>5Da%C=bd2nTOE^}jHY+-V9l#B&NY;R$7l#`Tb3r>f"
                "WeUvDS1a4t%Wt5BuW^8X^bT%}Uj0j?Bba!cFbZC@IK#T=nV{c?-luI0Shm=y3j0A3EX>F8C0*nT1VR>(Lbd-z%Lz"
                "HL~j1^&Wa$$KWW^8X^bT%|BATneuAVVpXXctX=0d<FzQj~?1E&"
            ),
        },
        {
            "text": ("(Array(float64, 2, 'F', False, aligned=True),)"),
            "pickle": (
                "fCQB`0RR910001t8E$oLVqq?0Z*pZWba`-Pb1q?Gb98cHV|0{^6JK<BaAjX|Wn*t{b98cbV{~tFl#`T<2V-w=d2("
                "fEl#C8va%E$0ZgX^Ubz^jIa+H&lj2UiqZDL_AV{dY0E_8WtWpgfWaCvljaAk9pj0Hh*a$$Ltla!1IVs&Y3bZKsLl"
                "#B*%Vrpe$bd-~nPKT6zlqie^WOR9OWt3<FXa$TIZgp*9VJ>5Da%C=bd2nTOE^}jHY+-V9l#B&NY;R$7l#`Tb3r>f"
                "WeUvDS1a4t%Wt5BuW^8X^bT%}Uj0j?Bba!cFbZC@IK#T=nV{c?-luI0Shm=y3j0A3EX>F8C0*nT1VR>(Lbd-z%Mw"
                "Dn2j1^&Wa$$KWW^8X^bT%|BATneuAVw*aXctZH0(FO!Qj~?1E&"
            ),
        },
    ),
    "haplotype_reconstruction.science.assembly.join.objective.forward": (
        {
            "text": (
                "(Array(float64, 1, 'C', False, aligned=True), Array(int64, 1, 'C', False, aligned=True),"
                " Array(int64, 1, 'C', False, aligned=True), Array(int64, 1, 'C', False, aligned=True), A"
                "rray(int64, 2, 'C', False, aligned=True), Array(int64, 1, 'C', False, aligned=True), Arr"
                "ay(int64, 1, 'C', False, aligned=True), float64, int64, Array(float64, 1, 'C', False, al"
                "igned=True), Array(bool, 1, 'C', False, aligned=True), Array(bool, 2, 'C', False, aligne"
                "d=True), Literal[bool](False), Literal[bool](False), Array(float64, 2, 'C', False, align"
                "ed=True))"
            ),
            "pickle": (
                "fCQCS0{{R30000ej2UiqZDL_AV{dY0E_8WtWpgfJVsmtIVPkZZj1ymUd2nT4a%E$0ZgX^Ubz^jIa+H&lj0a<HaCv"
                "fNXOxT%Uvgz*Z*FsRa&=>LZ*r8Al#Cf}b!}o{E@N+UWiE7iaAk8YZg6>Yd2nTOl#B&Ia&lpLl#`T<2x4_<Y;<XEb"
                "Ciq*Z(?d?V|0|0lun0~eUvDS1!Q!2aAlNe0%!${7;bfKVqq?0Z*pZWba`-Pb1rjZVQgV?bCiq)Mr?0kbd-~nXbVn"
                "<lzo&aj0A3BZDo{<2WD(<VRSY$l#B>sX>@mKWOQhhOF)bTUt@1%Wt2-Cb%&Hvl#B#!WNB@bO96}qY+-qCb##=B0Y"
                "j8%6pR&Na&lpLC}wPLVRSY$EFdvtEFeQElxP=Cf&q1hlv0#v0%!$j2xtpVhm?JkC}<970%!$j4~z#%Zggd5Wpb30"
                "lxPc1hm?JkC}<Rn1!-<{HZ+uI7E3^k26Jg=Ze?VYh-eo}8g+-1Qj};KO95ybXdGx1j1ysUa$$KWX>N2jG%O%7WGo"
                ";<DU@gzO@je-hm=y3Xe?+fXaZ;jXb5NvPKT6zlqhHpXenqKO9E&cXdGx1j1ysUa$$KWX>N2jG%O%8WGo;<DU@gzO"
                "^5+?hm=y3Xe?+fXc=fJXdh?-Xa#5pXbVn<lzo&aXbxxsXa#5wj0Zw*Z){~@Zj_UhXbVn<lzo&aXcUYDVsCG3lxP="
                "A5_N}^Qj};KO95ybXdGx1j1pmTa$$KWVsCG3EFdvtEFeQElxP=CsR4C|lv0#v0%!$j2xtpVhm?JkC}<97I%pb80%"
                "#j(9B34b5@B+3VR<NGZ*OcYATneuAVVpXXctX>0d<FzQj}-{Xa#5wj1EF?Z){~@ZcJ%(WpZI`l#`Tb3r>fWeUvDS"
                "4qt3(bY*g3Y+rU^Y;|RniHrnaYh`(qiHrnpVQpoUj1)|1bY*g3Y+GV)Z){yCMqzAoWhs=51z%%tWM!01y#b69Uu<"
                "b~WpZI`UvznJWnW`qV`ycRXgYO=lv0#vPG|yX1!xFp3r>fWeUvC@4rm!@8cPCb8)zJ86pR&Na&lpLC}wPLVRSY$E"
                "FdyuEFeQElxP=CeF1fclv0#*lr8"
            ),
        },
        {
            "text": (
                "(Array(float64, 1, 'C', False, aligned=True), Array(int64, 1, 'C', False, aligned=True),"
                " Array(int64, 1, 'C', False, aligned=True), Array(int64, 1, 'C', False, aligned=True), A"
                "rray(int64, 2, 'C', False, aligned=True), Array(int64, 1, 'C', False, aligned=True), Arr"
                "ay(int64, 1, 'C', False, aligned=True), float64, int64, Array(float64, 1, 'C', False, al"
                "igned=True), Array(bool, 1, 'C', False, aligned=True), Array(bool, 2, 'C', False, aligne"
                "d=True), Literal[bool](True), Literal[bool](False), Array(float64, 2, 'C', False, aligne"
                "d=True))"
            ),
            "pickle": (
                "fCQC~0{{R30000ej2UiqZDL_AV{dY0E_8WtWpgfJVsmtIVPkZZj1ymUd2nT4a%E$0ZgX^Ubz^jIa+H&lj0a<HaCv"
                "fNXOxT%Uvgz*Z*FsRa&=>LZ*r8Al#Cf}b!}o{E@N+UWiE7iaAk8YZg6>Yd2nTOl#B&Ia&lpLl#`T<2x4_<Y;<XEb"
                "Ciq*Z(?d?V|0|0lun0~eUvDS1!Q!2aAlNe0%!${7;bfKVqq?0Z*pZWba`-Pb1rjZVQgV?bCiq)Mr?0kbd-~nXbVn"
                "<lzo&aj0A3BZDo{<2WD(<VRSY$l#B>sX>@mKWOQhhOF)bTUt@1%Wt2-Cb%&Hvl#B#!WNB@bO96}qY+-qCb##=B0Y"
                "j8%6pR&Na&lpLC}wPLVRSY$EFdvtEFeQElxP=Cf&q1hlv0#v0%!$j2xtpVhm?JkC}<970%!$j4~z#%Zggd5Wpb30"
                "lxPc1hm?JkC}<Rn1!-<{HZ+uI7E3^k26Jg=Ze?VYh-eo}8g+-1Qj};KO95ybXdGx1j1ysUa$$KWX>N2jG%O%7WGo"
                ";<DU@gzO@je-hm=y3Xe?+fXaZ;jXb5NvPKT6zlqhHpXenqKO9E&cXdGx1j1ysUa$$KWX>N2jG%O%8WGo;<DU@gzO"
                "^5+?hm=y3Xe?+fXc=fJXdh?-Xa#5pXbVn<lzo&aXbxxsXa#5wj0Zw*Z){~@Zj_UhXbVn<lzo&aXcUYDVsCG3lxP="
                "A5_N}^Qj};KO95ybXdGx1j1pmTa$$KWVsCG3EFdvtEFeQElxP=CsR4C|lv0#v0%!$j2xtpVhm?JkC}<97I%pb80%"
                "#j(9B34b5@B+3VR<NGZ*OcYATneuAVVpXXctX>0d<FzQj}-{Xa#5wj1EF?Z){~@ZcJ%(WpZI`l#`Tb3r>fWeUvDS"
                "4qt3(bY*g3Y+rU^Y;|Rnh>QeZYh`(qh>QeoVQpoUj1x?0bY*g3Y+GV)Z){yCRC0A?DU^%_Ut@1%Wt2_60gMx0Y-x"
                "05a$#&=ba`-PUt?ioXl0aWI(3JXQj}-{Xa#6QXbVn<lzo&aXh(@?NQr1kj1)|1bY*g3Y+GV)Z){yCMqzAoWhs<sO"
                "HI82XiR82b%&HvlxPBI1!xFp3r>fWeUvC@4rm!@8cPCb8)zJ86pR&Na&lpLC}wPLVRSY$EFdyuEFeQElxP=CeF1f"
                "clv0#*lr8"
            ),
        },
        {
            "text": (
                "(Array(float64, 1, 'C', False, aligned=True), Array(int64, 1, 'C', False, aligned=True),"
                " Array(int64, 1, 'C', False, aligned=True), Array(int64, 1, 'C', False, aligned=True), A"
                "rray(int64, 2, 'C', False, aligned=True), Array(int64, 1, 'C', False, aligned=True), Arr"
                "ay(int64, 1, 'C', False, aligned=True), float64, int64, Array(float64, 1, 'C', False, al"
                "igned=True), Array(bool, 1, 'C', False, aligned=True), Array(bool, 2, 'C', False, aligne"
                "d=True), Literal[bool](True), Literal[bool](True), Array(float64, 2, 'C', False, aligned"
                "=True))"
            ),
            "pickle": (
                "fCQCR0{{R30000ej2UiqZDL_AV{dY0E_8WtWpgfJVsmtIVPkZZj1ymUd2nT4a%E$0ZgX^Ubz^jIa+H&lj0a<HaCv"
                "fNXOxT%Uvgz*Z*FsRa&=>LZ*r8Al#Cf}b!}o{E@N+UWiE7iaAk8YZg6>Yd2nTOl#B&Ia&lpLl#`T<2x4_<Y;<XEb"
                "Ciq*Z(?d?V|0|0lun0~eUvDS1!Q!2aAlNe0%!${7;bfKVqq?0Z*pZWba`-Pb1rjZVQgV?bCiq)Mr?0kbd-~nXbVn"
                "<lzo&aj0A3BZDo{<2WD(<VRSY$l#B>sX>@mKWOQhhOF)bTUt@1%Wt2-Cb%&Hvl#B#!WNB@bO96}qY+-qCb##=B0Y"
                "j8%6pR&Na&lpLC}wPLVRSY$EFdvtEFeQElxP=Cf&q1hlv0#v0%!$j2xtpVhm?JkC}<970%!$j4~z#%Zggd5Wpb30"
                "lxPc1hm?JkC}<Rn1!-<{HZ+uI7E3^k26Jg=Ze?VYh-eo}8g+-1Qj};KO95ybXdGx1j1ysUa$$KWX>N2jG%O%7WGo"
                ";<DU@gzO@je-hm=y3Xe?+fXaZ;jXb5NvPKT6zlqhHpXenqKO9E&cXdGx1j1ysUa$$KWX>N2jG%O%8WGo;<DU@gzO"
                "^5+?hm=y3Xe?+fXc=fJXdh?-Xa#5pXbVn<lzo&aXbxxsXa#5wj0Zw*Z){~@Zj_UhXbVn<lzo&aXcUYDVsCG3lxP="
                "A5_N}^Qj};KO95ybXdGx1j1pmTa$$KWVsCG3EFdvtEFeQElxP=CsR4C|lv0#v0%!$j2xtpVhm?JkC}<97I%pb80%"
                "#j(9B34b5@B+3VR<NGZ*OcYATneuAVVpXXctX>0d<FzQj}-{Xa#5wj1EF?Z){~@ZcJ%(WpZI`l#`Tb3r>fWeUvDS"
                "4qt3(bY*g3Y+rU^Y;|Rnh>QeZYh`(qh>QeoVQpoUj1x?0bY*g3Y+GV)Z){yCRC0A?DU^%_Ut@1%Wt2_60gMx0Y-x"
                "05a$#&=ba`-PUt?ioXl0aWI(3JXQj};;XaZ;jXb5NvPKT6zlqhHpXc=f4O9E&cXdGx1j1^&Wa$$KWW^8X^bT%|BA"
                "TneuAVVpXXctX=0d<FzQj~O*E&"
            ),
        },
        {
            "text": (
                "(Array(float64, 1, 'C', False, aligned=True), Array(int64, 1, 'C', False, aligned=True),"
                " Array(int64, 1, 'C', False, aligned=True), Array(int64, 1, 'C', False, aligned=True), A"
                "rray(int64, 2, 'C', False, aligned=True), Array(int64, 1, 'C', False, aligned=True), Arr"
                "ay(int64, 1, 'C', False, aligned=True), float64, uint64, Array(float64, 1, 'C', False, a"
                "ligned=True), Array(bool, 1, 'C', False, aligned=True), Array(bool, 2, 'C', False, align"
                "ed=True), Literal[bool](False), Literal[bool](False), Array(float64, 2, 'C', False, alig"
                "ned=True))"
            ),
            "pickle": (
                "fCQC(0{{R30000ej2UiqZDL_AV{dY0E_8WtWpgfJVsmtIVPkZZj1ymUd2nT4a%E$0ZgX^Ubz^jIa+H&lj0a<HaCv"
                "fNXOxT%Uvgz*Z*FsRa&=>LZ*r8Al#Cf}b!}o{E@N+UWiE7iaAk8YZg6>Yd2nTOl#B&Ia&lpLl#`T<2x4_<Y;<XEb"
                "Ciq*Z(?d?V|0|0lun0~eUvDS1!Q!2aAlNe0%!${7;bfKVqq?0Z*pZWba`-Pb1rjZVQgV?bCiq)Mr?0kbd-~nXbVn"
                "<lzo&aj0A3BZDo{<2WD(<VRSY$l#B>sX>@mKWOQhhOF)bTUt@1%Wt2-Cb%&Hvl#B#!WNB@bO96}qY+-qCb##=B0Y"
                "j8%6pR&Na&lpLC}wPLVRSY$EFdvtEFeQElxP=Cf&q1hlv0#v0%!$j2xtpVhm?JkC}<970%!$j4~z#%Zggd5Wpb30"
                "lxPc1hm?JkC}<Rn1!-<{HZ+uI7E3^k26Jg=Ze?VYh-eo}8g+-1Qj};KO95ybXdGx1j1ysUa$$KWX>N2jG%O%7WGo"
                ";<DU@gzO@je-hm=y3Xe?+fXaZ;jXb5NvPKT6zlqhHpXenqKO9E&cXdGx1j1ysUa$$KWX>N2jG%O%8WGo;<DU@gzO"
                "^5+?hm=y3Xe?+fXc=e%Xa#5^XbVn<lzo&aXcUYFb!l#NHZ+uI7E3^ACy8hmOBQv9lv0#vA7}z-1!xFp3r>fWeUvC"
                "@4rl^s1!xb92SRUeY-M3?l#`Tb3r>fWeUvC@6pRF7Z*OdrXctQob%&HvlxP}D0caa&9B34b5@B+3VR<NGZ*OcYAT"
                "eYtAVVpXXctYX0d<FzQj}-{Xa#5pXbVn<lzo&aXbxyUXc|ibXd7r8XcUYRVRCX|c_?CUZ)_|eGGr_uLn)MK7fpTv"
                "b%&HvlxPBI1!xb94nl8lY-M3?Olfpwa$#(glay!+PKT6zlqif2Uu<b~WpZI`Uv^<^b!C)^j09h6WqFi|j0A3BZDo"
                "{<6ijJ!WpZI`TVii-Y+WcuVQh0{DU^%_Ut@1%Wt2_50gMx0Y-x05a$#&=ba`-PUt?ioXl0aWKXr$cQj};@XaZ;jX"
                "b5NvPKT6zlqhHpXc=f4O9E&cXdGx1j1^&Wa$$KWW^8X^bT%|BATneuAVVpXXctX=0d<FzQj~O*E&"
            ),
        },
        {
            "text": (
                "(Array(float64, 1, 'C', False, aligned=True), Array(int64, 1, 'C', False, aligned=True),"
                " Array(int64, 1, 'C', False, aligned=True), Array(int64, 1, 'C', False, aligned=True), A"
                "rray(int64, 2, 'C', False, aligned=True), Array(int64, 1, 'C', False, aligned=True), Arr"
                "ay(int64, 1, 'C', False, aligned=True), float64, uint64, Array(float64, 1, 'C', False, a"
                "ligned=True), Array(bool, 1, 'C', False, aligned=True), Array(bool, 2, 'C', False, align"
                "ed=True), Literal[bool](True), Literal[bool](False), Array(float64, 2, 'C', False, align"
                "ed=True))"
            ),
            "pickle": (
                "fCQDc0{{R30000ej2UiqZDL_AV{dY0E_8WtWpgfJVsmtIVPkZZj1ymUd2nT4a%E$0ZgX^Ubz^jIa+H&lj0a<HaCv"
                "fNXOxT%Uvgz*Z*FsRa&=>LZ*r8Al#Cf}b!}o{E@N+UWiE7iaAk8YZg6>Yd2nTOl#B&Ia&lpLl#`T<2x4_<Y;<XEb"
                "Ciq*Z(?d?V|0|0lun0~eUvDS1!Q!2aAlNe0%!${7;bfKVqq?0Z*pZWba`-Pb1rjZVQgV?bCiq)Mr?0kbd-~nXbVn"
                "<lzo&aj0A3BZDo{<2WD(<VRSY$l#B>sX>@mKWOQhhOF)bTUt@1%Wt2-Cb%&Hvl#B#!WNB@bO96}qY+-qCb##=B0Y"
                "j8%6pR&Na&lpLC}wPLVRSY$EFdvtEFeQElxP=Cf&q1hlv0#v0%!$j2xtpVhm?JkC}<970%!$j4~z#%Zggd5Wpb30"
                "lxPc1hm?JkC}<Rn1!-<{HZ+uI7E3^k26Jg=Ze?VYh-eo}8g+-1Qj};KO95ybXdGx1j1ysUa$$KWX>N2jG%O%7WGo"
                ";<DU@gzO@je-hm=y3Xe?+fXaZ;jXb5NvPKT6zlqhHpXenqKO9E&cXdGx1j1ysUa$$KWX>N2jG%O%8WGo;<DU@gzO"
                "^5+?hm=y3Xe?+fXc=e%Xa#5^XbVn<lzo&aXcUYFb!l#NHZ+uI7E3^ACy8hmOBQv9lv0#vA7}z-1!xFp3r>fWeUvC"
                "@4rl^s1!xb92SRUeY-M3?l#`Tb3r>fWeUvC@6pRF7Z*OdrXctQob%&HvlxP}D0caa&9B34b5@B+3VR<NGZ*OcYAT"
                "eYtAVVpXXctYX0d<FzQj}-{Xa#5pXbVn<lzo&aXbxyUXc|ibXd7r8XcUYRVRCX|c_?CUZ)_|eGGr_uLn)MK7fpTv"
                "b%&HvlxPBI1!xb94nl8lY-M3?Olfpwa$#(glay!+PKT6zlqif2Uu<b~WpZI`Uv^<^b!C)@j09h6WqFi{j0A3BZDo"
                "{<6HIAzWpZI`TVii-Y+Wc+a&=`Xl#B&mV{c?-luf<?j1ymMX>?_BVQgP?d2nT4V_{=xWt3<?b%&HvlxPBI1!zfV3"
                "r>fWeUvC@Oo?bsiD*uY6ijJ!WpZI`TVii-Y+WcuVQh0{DU@hXO}znVQD{GPhm=y3XaZ;jXb5NvPKT6zlqhHpXc=f"
                "4O9E&cXdGx1j1^&Wa$$KWW^8X^bT%|BATneuAVVpXXctX=0d<FzQj~O*E&"
            ),
        },
        {
            "text": (
                "(Array(float64, 1, 'C', False, aligned=True), Array(int64, 1, 'C', False, aligned=True),"
                " Array(int64, 1, 'C', False, aligned=True), Array(int64, 1, 'C', False, aligned=True), A"
                "rray(int64, 2, 'C', False, aligned=True), Array(int64, 1, 'C', False, aligned=True), Arr"
                "ay(int64, 1, 'C', False, aligned=True), float64, uint64, Array(float64, 1, 'C', False, a"
                "ligned=True), Array(bool, 1, 'C', False, aligned=True), Array(bool, 2, 'C', False, align"
                "ed=True), Literal[bool](True), Literal[bool](True), Array(float64, 2, 'C', False, aligne"
                "d=True))"
            ),
            "pickle": (
                "fCQC&0{{R30000ej2UiqZDL_AV{dY0E_8WtWpgfJVsmtIVPkZZj1ymUd2nT4a%E$0ZgX^Ubz^jIa+H&lj0a<HaCv"
                "fNXOxT%Uvgz*Z*FsRa&=>LZ*r8Al#Cf}b!}o{E@N+UWiE7iaAk8YZg6>Yd2nTOl#B&Ia&lpLl#`T<2x4_<Y;<XEb"
                "Ciq*Z(?d?V|0|0lun0~eUvDS1!Q!2aAlNe0%!${7;bfKVqq?0Z*pZWba`-Pb1rjZVQgV?bCiq)Mr?0kbd-~nXbVn"
                "<lzo&aj0A3BZDo{<2WD(<VRSY$l#B>sX>@mKWOQhhOF)bTUt@1%Wt2-Cb%&Hvl#B#!WNB@bO96}qY+-qCb##=B0Y"
                "j8%6pR&Na&lpLC}wPLVRSY$EFdvtEFeQElxP=Cf&q1hlv0#v0%!$j2xtpVhm?JkC}<970%!$j4~z#%Zggd5Wpb30"
                "lxPc1hm?JkC}<Rn1!-<{HZ+uI7E3^k26Jg=Ze?VYh-eo}8g+-1Qj};KO95ybXdGx1j1ysUa$$KWX>N2jG%O%7WGo"
                ";<DU@gzO@je-hm=y3Xe?+fXaZ;jXb5NvPKT6zlqhHpXenqKO9E&cXdGx1j1ysUa$$KWX>N2jG%O%8WGo;<DU@gzO"
                "^5+?hm=y3Xe?+fXc=e%Xa#5^XbVn<lzo&aXcUYFb!l#NHZ+uI7E3^ACy8hmOBQv9lv0#vA7}z-1!xFp3r>fWeUvC"
                "@4rl^s1!xb92SRUeY-M3?l#`Tb3r>fWeUvC@6pRF7Z*OdrXctQob%&HvlxP}D0caa&9B34b5@B+3VR<NGZ*OcYAT"
                "eYtAVVpXXctYX0d<FzQj}-{Xa#5pXbVn<lzo&aXbxyUXc|ibXd7r8XcUYRVRCX|c_?CUZ)_|eGGr_uLn)MK7fpTv"
                "b%&HvlxPBI1!xb94nl8lY-M3?Olfpwa$#(glay!+PKT6zlqif2Uu<b~WpZI`Uv^<^b!C)@j09h6WqFi{j0A3BZDo"
                "{<6HIAzWpZI`TVii-Y+Wc+a&=`Xl#B&mV{c?-luf<?j1ymMX>?_BVQgP?d2nT4V_{=xWt3<?b%&HvlxS0E0%!$j2"
                "xtpVhm?JkC}<978E6_y0%#j(9B34b6=8C6VR<NKY;R$7HZ&|CGGr_uLn)MK7fpQub%&HvlysCX"
            ),
        },
    ),
    "haplotype_reconstruction.science.assembly.join.objective.minimum_switches": (
        {
            "text": (
                "(Array(float64, 1, 'C', False, aligned=True), Array(int64, 1, 'C', False, aligned=True),"
                " Array(int64, 1, 'C', False, aligned=True), Array(int64, 1, 'C', False, aligned=True), A"
                "rray(int64, 2, 'C', False, aligned=True), float64, int64)"
            ),
            "pickle": (
                "fCQEB0RR910000ej2UiqZDL_AV{dY0E_8WtWpgfJVsmtIVPkZZj1ymUd2nT4a%E$0ZgX^Ubz^jIa+H&lj0a<HaCv"
                "fNXOxT%Uvgz*Z*FsRa&=>LZ*r8Al#Cf}b!}o{E@N+UWiE7iaAk8YZg6>Yd2nTOl#B&Ia&lpLl#`T<2x4_<Y;<XEb"
                "Ciq*Z(?d?V|0|0lun0~eUvDS1!Q!2aAlNe0%!${7;bfKVqq?0Z*pZWba`-Pb1rjZVQgV?bCiq)Mr?0kbd-~nXbVn"
                "<lzo&aj0A3BZDo{<2WD(<VRSY$l#B>sX>@mKWOQhhOF)bTUt@1%Wt2-Cb%&Hvl#B#!WNB@bO96}qY+-qCb##=B0Y"
                "j8%6pR&Na&lpLC}wPLVRSY$EFdvtEFeQElxP=Cf&q1hlv0#v0%!$j2xtpVhm?JkC}<970%!$j4~z#%Zggd5Wpb30"
                "lxPc1hm?JkC}<Rn1!-<{HZ+uI7E3^k26Jg=Ze?VYh-eo}8g+-1Qj};KO95ybXdGx1j1ysUa$$KWX>N2jG%O%7WGo"
                ";<DU@gzO@je-hm=y3Xe?+fXaZ;jXb5NvPKT6zlqhHpXenqKO9E&cXdGx1j1ysUa$$KWX>N2jG%O%8WGo;<DU@gzP"
                "1XW+hm=y3Xc=fJbd)X"
            ),
        },
    ),
    "haplotype_reconstruction.science.assembly.join.objective.paint": (
        {
            "text": (
                "(Array(float64, 1, 'C', False, aligned=True), Array(int64, 1, 'C', False, aligned=True),"
                " Array(int64, 1, 'C', False, aligned=True), Array(int64, 1, 'C', False, aligned=True), A"
                "rray(int64, 2, 'C', False, aligned=True), float64, int64, int64, float64, bool)"
            ),
            "pickle": (
                "fCQBw0ssI20000ej2UiqZDL_AV{dY0E_8WtWpgfJVsmtIVPkZZj1ymUd2nT4a%E$0ZgX^Ubz^jIa+H&lj0a<HaCv"
                "fNXOxT%Uvgz*Z*FsRa&=>LZ*r8Al#Cf}b!}o{E@N+UWiE7iaAk8YZg6>Yd2nTOl#B&Ia&lpLl#`T<2x4_<Y;<XEb"
                "Ciq*Z(?d?V|0|0lun0~eUvDS1!Q!2aAlNe0%!${7;bfKVqq?0Z*pZWba`-Pb1rjZVQgV?bCiq)Mr?0kbd-~nXbVn"
                "<lzo&aj0A3BZDo{<2WD(<VRSY$l#B>sX>@mKWOQhhOF)bTUt@1%Wt2-Cb%&Hvl#B#!WNB@bO96}qY+-qCb##=B0Y"
                "j8%6pR&Na&lpLC}wPLVRSY$EFdvtEFeQElxP=Cf&q1hlv0#v0%!$j2xtpVhm?JkC}<970%!$j4~z#%Zggd5Wpb30"
                "lxPc1hm?JkC}<Rn1!-<{HZ+uI7E3^k26Jg=Ze?VYh-eo}8g+-1Qj};KO95ybXdGx1j1ysUa$$KWX>N2jG%O%7WGo"
                ";<DU@gzO@je-hm=y3Xe?+fXaZ;jXb5NvPKT6zlqhHpXenqKO9E&cXdGx1j1ysUa$$KWX>N2jG%O%8WGo;<DU@gzO"
                "^5+?hm=y3Xc=fJXenqJXaZ;jXb+4BLT_(uWnpfVlay!+PKT6zlqhHvj09qDZ)}uk7fTX#hm=y3bd)X"
            ),
        },
    ),
    "haplotype_reconstruction.science.assembly.join.objective.pair_lists": (
        {
            "text": ("(Array(bool, 1, 'C', False, aligned=True),)"),
            "pickle": (
                "fCQB#0RR910001t8E$oLVqq?0Z*pZWba`-Pb1q?Gb98cHV|0{^6JK<BaAjX|Wn*t{b98cbV{~tFl#`T<2V-w=d2("
                "fEl#C8va%E$0ZgX^Ubz^jIa+H&lj2UiqZDL_AV{dY0E_8WtWpgfWaCvljaAk9pj0Hh*a$$Ltla!1IVs&Y3bZKsLl"
                "#B*%Vrpe$bd-~nPKT6zlqie^WOR9OWt3<FXa$TIZgp*9VJ>5Da%C=bd2nTOE^}jHY+-V9l#B;LZ*Od6VQ!R@lxPc"
                "1hm?JkD2xPdVQpoUj09qDZ)}u|1z%%tWMz~~5_N}^Qk0AYZe(d~luH4O25ez@Z*_E(i~&QGXcUYRVRCX|c_?CUZ)"
                "_|eF=Q+tLn)MK7EOu)b%&Hvl!cTo"
            ),
        },
    ),
    "haplotype_reconstruction.science.assembly.join.objective.record_covers": (
        {
            "text": (
                "(Array(float64, 1, 'C', False, aligned=True), float64, Array(int64, 1, 'C', False, align"
                "ed=True), Array(int64, 1, 'C', False, aligned=True), Array(bool, 1, 'C', False, aligned="
                "True), Array(bool, 2, 'C', False, aligned=True), Literal[bool](False))"
            ),
            "pickle": (
                "fCQD%0ssI20000ej2UiqZDL_AV{dY0E_8WtWpgfJVsmtIVPkZZj1ymUd2nT4a%E$0ZgX^Ubz^jIa+H&lj0a<HaCv"
                "fNXOxT%Uvgz*Z*FsRa&=>LZ*r8Al#Cf}b!}o{E@N+UWiE7iaAk8YZg6>Yd2nTOl#B&Ia&lpLl#`T<2x4_<Y;<XEb"
                "Ciq*Z(?d?V|0|0lun0~eUvDS1!Q!2aAlNe0%!${7;bfKVqq?0Z*pZWba`-Pb1rjZVQgV?bCiq)Mr?0kbd-~nXbVn"
                "<lzo&aj0A3BZDo{<2WD(<VRSY$l#B>sX>@mKWOQhhOF)bTUt@1%Wt2-Cb%&Hvl#B#!WNB@bO96}qY+-qCb##=B0Y"
                "j8%6pR&Na&lpLC}wPLVRSY$EFdvtEFeQElxP=Cf&q1hlv0#v8E67%1!xFp3r>fWeUvC@4rl^s1!xb92T5*pWoKn_"
                "l#`Tb3r>fWeUvC@6pRIFZge&@lxP-9K#T@+X=iR_WR!?#7fTv-hm=y3Xc|iaXd7r8XcUYSVRCX|c_?XabT%|BATe"
                "YtAVVpXXctX`0d<FzQj};cXaZ;jXb5NvPKT6zlqhHpXaZ;jXb+4BLT_(uWnpfVlay!+PKT6zlqhHvj09qDZ)}uk7"
                "fTX#hm=y3Xc|iaXd7r8XcUYRVRCX|c_?CUZ)_|eF=Q+tLn)MK7fq=Fb%&HvlxPBI1!xFp3r>fWeUvC@4rn!K8cPC"
                "b8)zJ86pRvKa&lpLC}MAKY%Cx$WGo;<DU@gzO@0A&hm=y3XaZ;jXb+4ILT_(uWnpeiX>?_BVQiF>lxPc1hm?JkD2"
                "xtYY-x05a$#&=c42IFWt54G1Yc`qd6bEa1a4t%Wt5B*Olfpwa$#&+VsCG3T_{FjY;$EPl#B&mV{c?-luf+>j1ymM"
                "X>?_BVQgP?d2nT4V_{=xWt3<&b%&HvlysCX"
            ),
        },
        {
            "text": (
                "(Array(float64, 1, 'C', False, aligned=True), float64, Array(int64, 1, 'C', False, align"
                "ed=True), Array(int64, 1, 'C', False, aligned=True), Array(bool, 1, 'C', False, aligned="
                "True), Array(bool, 2, 'C', False, aligned=True), Literal[bool](True))"
            ),
            "pickle": (
                "fCQD$0ssI20000ej2UiqZDL_AV{dY0E_8WtWpgfJVsmtIVPkZZj1ymUd2nT4a%E$0ZgX^Ubz^jIa+H&lj0a<HaCv"
                "fNXOxT%Uvgz*Z*FsRa&=>LZ*r8Al#Cf}b!}o{E@N+UWiE7iaAk8YZg6>Yd2nTOl#B&Ia&lpLl#`T<2x4_<Y;<XEb"
                "Ciq*Z(?d?V|0|0lun0~eUvDS1!Q!2aAlNe0%!${7;bfKVqq?0Z*pZWba`-Pb1rjZVQgV?bCiq)Mr?0kbd-~nXbVn"
                "<lzo&aj0A3BZDo{<2WD(<VRSY$l#B>sX>@mKWOQhhOF)bTUt@1%Wt2-Cb%&Hvl#B#!WNB@bO96}qY+-qCb##=B0Y"
                "j8%6pR&Na&lpLC}wPLVRSY$EFdvtEFeQElxP=Cf&q1hlv0#v8E67%1!xFp3r>fWeUvC@4rl^s1!xb92T5*pWoKn_"
                "l#`Tb3r>fWeUvC@6pRIFZge&@lxP-9K#T@+X=iR_WR!?#7fTv-hm=y3Xc|iaXd7r8XcUYSVRCX|c_?XabT%|BATe"
                "YtAVVpXXctX`0d<FzQj};cXaZ;jXb5NvPKT6zlqhHpXaZ;jXb+4BLT_(uWnpfVlay!+PKT6zlqhHvj09qDZ)}uk7"
                "fTX#hm=y3Xc|iaXd7r8XcUYRVRCX|c_?CUZ)_|eF=Q+tLn)MK7fq=Fb%&HvlxPBI1!xFp3r>fWeUvC@4rn!K8cPC"
                "b8)zJ86pRvKa&lpLC}MAKY%Cx$WGo;<DU@gzO@0A&hm=y3XaZ;jXb+4ILT_(uWnpeiX>?_BVQiF>lxPc1hm?JkD2"
                "xtYY-x05a$#&=c42IFWt51F1Yc`qd6bBZ1a4t%Wt5B)Olfpwa$#&+VsCG3T_{v?b!91(j0Im~Z)9bZO}+t)6JKm;"
                "bY*g3Y+rPFaAjX(VPj}zlxQ_|hm=y3bd)X"
            ),
        },
    ),
    "haplotype_reconstruction.science.assembly.join.objective.score_samples": (
        {
            "text": (
                "(Array(float64, 1, 'C', False, aligned=True), Array(int64, 1, 'C', False, aligned=True),"
                " Array(int64, 1, 'C', False, aligned=True), Array(int64, 1, 'C', False, aligned=True), A"
                "rray(int64, 2, 'C', False, aligned=True), float64, Array(int64, 1, 'C', False, aligned=T"
                "rue))"
            ),
            "pickle": (
                "fCQEB0RR910000ej2UiqZDL_AV{dY0E_8WtWpgfJVsmtIVPkZZj1ymUd2nT4a%E$0ZgX^Ubz^jIa+H&lj0a<HaCv"
                "fNXOxT%Uvgz*Z*FsRa&=>LZ*r8Al#Cf}b!}o{E@N+UWiE7iaAk8YZg6>Yd2nTOl#B&Ia&lpLl#`T<2x4_<Y;<XEb"
                "Ciq*Z(?d?V|0|0lun0~eUvDS1!Q!2aAlNe0%!${7;bfKVqq?0Z*pZWba`-Pb1rjZVQgV?bCiq)Mr?0kbd-~nXbVn"
                "<lzo&aj0A3BZDo{<2WD(<VRSY$l#B>sX>@mKWOQhhOF)bTUt@1%Wt2-Cb%&Hvl#B#!WNB@bO96}qY+-qCb##=B0Y"
                "j8%6pR&Na&lpLC}wPLVRSY$EFdvtEFeQElxP=Cf&q1hlv0#v0%!$j2xtpVhm?JkC}<970%!$j4~z#%Zggd5Wpb30"
                "lxPc1hm?JkC}<Rn1!-<{HZ+uI7E3^k26Jg=Ze?VYh-eo}8g+-1Qj};KO95ybXdGx1j1ysUa$$KWX>N2jG%O%7WGo"
                ";<DU@gzO@je-hm=y3Xe?+fXaZ;jXb5NvPKT6zlqhHpXenqKO9E&cXdGx1j1ysUa$$KWX>N2jG%O%8WGo;<DU@gzO"
                "^5+?hm=y3Xc=fMbd)X"
            ),
        },
    ),
    "haplotype_reconstruction.science.assembly.join.objective.widest_block": (
        {
            "text": ("(Array(int64, 1, 'C', False, aligned=True),)"),
            "pickle": (
                "fCQC30RR910001t8E$oLVqq?0Z*pZWba`-Pb1q?Gb98cHV|0{^6JK<BaAjX|Wn*t{b98cbV{~tFl#`T<2V-w=d2("
                "fEl#C8va%E$0ZgX^Ubz^jIa+H&lj2UiqZDL_AV{dY0E_8WtWpgfWaCvljaAk9pj0Hh*a$$Ltla!1IVs&Y3bZKsLl"
                "#B*%Vrpe$bd-~nPKT6zlqie^WOR9OWt3<FXa$TIZgp*9VJ>5Da%C=bd2nTOE^}jHY+-V9l#B;SZggd5Wpb30lxPc"
                "1hm?JkD2xPdVQpoUj0I_KbT%}Uj0j?Bba!cFbZC@IK#T@+X=iR_WR!@E1z%%tWMz~~8g+-1Qk0AYZe(d~luH4O25"
                "ez@Z*_E(i~&QGXcUYSVRCX|c_?XabT%|BATeYtAVVpXXc$d{0d<FzQj~?1E&"
            ),
        },
    ),
    "haplotype_reconstruction.science.assembly.join.pair_segments.best_without": (
        {
            "text": (
                "(Array(float64, 1, 'C', False, aligned=True), Array(int64, 1, 'C', False, aligned=True),"
                " Array(int64, 1, 'C', False, aligned=True), int64, int64)"
            ),
            "pickle": (
                "fCQDY0RR910000ej2UiqZDL_AV{dY0E_8WtWpgfJVsmtIVPkZZj1ymUd2nT4a%E$0ZgX^Ubz^jIa+H&lj0a<HaCv"
                "fNXOxT%Uvgz*Z*FsRa&=>LZ*r8Al#Cf}b!}o{E@N+UWiE7iaAk8YZg6>Yd2nTOl#B&Ia&lpLl#`T<2x4_<Y;<XEb"
                "Ciq*Z(?d?V|0|0lun0~eUvDS1!Q!2aAlNe0%!${7;bfKVqq?0Z*pZWba`-Pb1rjZVQgV?bCiq)Mr?0kbd-~nXbVn"
                "<lzo&aj0A3BZDo{<2WD(<VRSY$l#B>sX>@mKWOQhhOF)bTUt@1%Wt2-Cb%&Hvl#B#!WNB@bO96}qY+-qCb##=B0Y"
                "j8%6pR&Na&lpLC}wPLVRSY$EFdvtEFeQElxP=Ce*txelv0#v0%!$j2xtpVhm?JkC}<970%!$j4~z#%Zggd5Wpb30"
                "lxPc1hm?JkC}<Rn1!-<{HZ+uI7E3^k26Jg=Ze?VYh-eo}8g+-1Qj};KO95ybXdGx1j1ysUa$$KWX>N2jG%O%7WGo"
                ";<DU@gzO_l+5hm=y3Xe?+cXeo4*E&"
            ),
        },
    ),
    "haplotype_reconstruction.science.assembly.join.pair_segments.choose_segments": (
        {
            "text": ("(int64, int64, Literal[int](0))"),
            "pickle": (
                "fCQCe0RR910001t8E$oLVqq?0Z*pZWba`-Pb1q?Gb98cHV|0{^6JK<BaAjX|Wn*t{b98cbV{~tFl#`T<2V-w=d2("
                "fEl#C8va%E$0ZgX^Ubz^jIa+H&lj2LcpZDL_AV{dY0E_8WtWpgfbV_|Gza&wf72T5*pWoKn_l#`T<2x4_<Y;<XEb"
                "Ciq*Z(?d?V|0|0lun0~eUvDS1a4t%Wt5BsX>N2jG?a`8Vrg`DX=HS0luJO226Jg=Ze?VYh>QhaV{c?-luH_Qhm=y"
                "3XcTAyXa#5nj1Ea|bY*8{a!hG-WpZI`l#`Tb3r>fWeUvDS4qt3(bY*g3Y+rU^Y;|RnO8|@nUu$J~luH1N6JKm;bY"
                "*g3Y+rPFaAjX(VPj}zlxP%;1a4t%Wt5B$Olfpwa$#&+X>N2~C@?9Mj0j?Bba!cFbZC@IK#T@+X=iR_WR!@E1z%%t"
                "WM!01%mH<Wlv0$3lr8"
            ),
        },
        {
            "text": ("(int64, int64, int64)"),
            "pickle": (
                "fCQD$000000001t8E$oLVqq?0Z*pZWba`-Pb1q?Gb98cHV|0{^6JK<BaAjX|Wn*t{b98cbV{~tFl#`T<2V-w=d2("
                "fEl#C8va%E$0ZgX^Ubz^jIa+H&lj2LcpZDL_AV{dY0E_8WtWpgfbV_|Gza&wf72T5*pWoKn_l#`T<2x4_<Y;<XEb"
                "Ciq*Z(?d?V|0|0lun0~eUvDS1a4t%Wt5BsX>N2jG?a`8Vrg`DX=HS0luJO226Jg=Ze?VYh>QhaV{c?-luH_Qhm=y"
                "3XcTA^hm<Y"
            ),
        },
    ),
    "haplotype_reconstruction.science.assembly.join.pair_segments.interval_values": (
        {
            "text": (
                "(Array(float64, 1, 'C', False, aligned=True), Array(int64, 1, 'C', False, aligned=True),"
                " Array(int64, 1, 'C', False, aligned=True), Array(int64, 1, 'C', False, aligned=True), A"
                "rray(int64, 1, 'C', False, aligned=True), Array(float64, 1, 'C', False, aligned=True))"
            ),
            "pickle": (
                "fCQDa0RR910000ej2UiqZDL_AV{dY0E_8WtWpgfJVsmtIVPkZZj1ymUd2nT4a%E$0ZgX^Ubz^jIa+H&lj0a<HaCv"
                "fNXOxT%Uvgz*Z*FsRa&=>LZ*r8Al#Cf}b!}o{E@N+UWiE7iaAk8YZg6>Yd2nTOl#B&Ia&lpLl#`T<2x4_<Y;<XEb"
                "Ciq*Z(?d?V|0|0lun0~eUvDS1!Q!2aAlNe0%!${7;bfKVqq?0Z*pZWba`-Pb1rjZVQgV?bCiq)Mr?0kbd-~nXbVn"
                "<lzo&aj0A3BZDo{<2WD(<VRSY$l#B>sX>@mKWOQhhOF)bTUt@1%Wt2-Cb%&Hvl#B#!WNB@bO96}qY+-qCb##=B0Y"
                "j8%6pR&Na&lpLC}wPLVRSY$EFdvtEFeQElxP=Cf&q1hlv0#v0%!$j2xtpVhm?JkC}<970%!$j4~z#%Zggd5Wpb30"
                "lxPc1hm?JkC}<Rn1!-<{HZ+uI7E3^k26Jg=Ze?VYh-eo}8g+-1Qj};KO95ybXdGx1j1ysUa$$KWX>N2jG%O%7WGo"
                ";<DU@gzO@je-hm=y3Xe?+fXe?+Sbd)X"
            ),
        },
    ),
    "haplotype_reconstruction.science.assembly.join.pair_segments.pair_segment_dp": (
        {
            "text": (
                "(Array(float64, 2, 'C', False, aligned=True), Array(int64, 1, 'C', False, aligned=True),"
                " Array(int64, 1, 'C', False, aligned=True), Array(int64, 1, 'C', False, aligned=True), A"
                "rray(int64, 1, 'C', False, aligned=True), float64, omitted(default=True))"
            ),
            "pickle": (
                "fCQBw0ssI20000ej2UiqZDL_AV{dY0E_8WtWpgfJVsmtIVPkZZj1ymUd2nT4a%E$0ZgX^Ubz^jIa+H&lj0a<HaCv"
                "fNXOxT%Uvgz*Z*FsRa&=>LZ*r8Al#Cf}b!}o{E@N+UWiE7iaAk8YZg6>Yd2nTOl#B&Ia&lpLl#`T<2x4_<Y;<XEb"
                "Ciq*Z(?d?V|0|0lun0~eUvDS1!Q!2aAlNe0%!${7;bfKVqq?0Z*pZWba`-Pb1rjZVQgV?bCiq)Mr?0kbd-~nXbVn"
                "<lzo&aj0A3BZDo{<2WD(<VRSY$l#B>sX>@mKWOQhhOF)bTUt@1%Wt2-Cb%&Hvl#B#!WNB@bO9G4rY+-qCb##=B0Y"
                "j8%6pR&Na&lpLC}wPLVRSY$EFdyuEFeQElxP=CeF1fclv0#v0%!$j2xtpVhm?JkC}<970%!$j4~z#%Zggd5Wpb30"
                "lxPc1hm?JkC}<Rn1!-<{HZ+uI7E3^k26Jg=Ze?VYh-eo}8g+-1Qj};KO95ybXdGx1j1ysUa$$KWX>N2jG%O%7WGo"
                ";<DU@gzO@je-hm=y3Xe?+fXe?+MXaZ;jj1_KmZDL_AV{dY0E_8WtWpgfVX>((gj0aC`X>@dDWR#PXXbVn<lzo&aj"
                "0RtJVQh6}l!%N9Uv^<^b!A^`WqFi{XcUYUZ*6IGbY)~HWMyVyb!>D!RC0A?DU@gzP09gvhm=y3bd)X"
            ),
        },
    ),
    "haplotype_reconstruction.science.assembly.join.pair_segments.regions": (
        {
            "text": (
                "(Array(float64, 1, 'C', False, aligned=True), Array(int64, 1, 'C', False, aligned=True),"
                " Array(int64, 1, 'C', False, aligned=True), Array(int64, 1, 'C', False, aligned=True), A"
                "rray(int64, 2, 'C', False, aligned=True), float64, int64, Array(int64, 1, 'C', False, al"
                "igned=True), Array(int64, 1, 'C', False, aligned=True))"
            ),
            "pickle": (
                "fCQEF0RR910000ej2UiqZDL_AV{dY0E_8WtWpgfJVsmtIVPkZZj1ymUd2nT4a%E$0ZgX^Ubz^jIa+H&lj0a<HaCv"
                "fNXOxT%Uvgz*Z*FsRa&=>LZ*r8Al#Cf}b!}o{E@N+UWiE7iaAk8YZg6>Yd2nTOl#B&Ia&lpLl#`T<2x4_<Y;<XEb"
                "Ciq*Z(?d?V|0|0lun0~eUvDS1!Q!2aAlNe0%!${7;bfKVqq?0Z*pZWba`-Pb1rjZVQgV?bCiq)Mr?0kbd-~nXbVn"
                "<lzo&aj0A3BZDo{<2WD(<VRSY$l#B>sX>@mKWOQhhOF)bTUt@1%Wt2-Cb%&Hvl#B#!WNB@bO96}qY+-qCb##=B0Y"
                "j8%6pR&Na&lpLC}wPLVRSY$EFdvtEFeQElxP=Cf&q1hlv0#v0%!$j2xtpVhm?JkC}<970%!$j4~z#%Zggd5Wpb30"
                "lxPc1hm?JkC}<Rn1!-<{HZ+uI7E3^k26Jg=Ze?VYh-eo}8g+-1Qj};KO95ybXdGx1j1ysUa$$KWX>N2jG%O%7WGo"
                ";<DU@gzO@je-hm=y3Xe?+fXaZ;jXb5NvPKT6zlqhHpXenqKO9E&cXdGx1j1ysUa$$KWX>N2jG%O%8WGo;<DU@gzO"
                "^5+?hm=y3Xc=fJXe?+fbd)X"
            ),
        },
        {
            "text": (
                "(Array(float64, 1, 'C', False, aligned=True), Array(int64, 1, 'C', False, aligned=True),"
                " Array(int64, 1, 'C', False, aligned=True), Array(int64, 1, 'C', False, aligned=True), A"
                "rray(int64, 2, 'C', False, aligned=True), float64, uint64, Array(int64, 1, 'C', False, a"
                "ligned=True), Array(int64, 1, 'C', False, aligned=True))"
            ),
            "pickle": (
                "fCQBr0ssI20000ej2UiqZDL_AV{dY0E_8WtWpgfJVsmtIVPkZZj1ymUd2nT4a%E$0ZgX^Ubz^jIa+H&lj0a<HaCv"
                "fNXOxT%Uvgz*Z*FsRa&=>LZ*r8Al#Cf}b!}o{E@N+UWiE7iaAk8YZg6>Yd2nTOl#B&Ia&lpLl#`T<2x4_<Y;<XEb"
                "Ciq*Z(?d?V|0|0lun0~eUvDS1!Q!2aAlNe0%!${7;bfKVqq?0Z*pZWba`-Pb1rjZVQgV?bCiq)Mr?0kbd-~nXbVn"
                "<lzo&aj0A3BZDo{<2WD(<VRSY$l#B>sX>@mKWOQhhOF)bTUt@1%Wt2-Cb%&Hvl#B#!WNB@bO96}qY+-qCb##=B0Y"
                "j8%6pR&Na&lpLC}wPLVRSY$EFdvtEFeQElxP=Cf&q1hlv0#v0%!$j2xtpVhm?JkC}<970%!$j4~z#%Zggd5Wpb30"
                "lxPc1hm?JkC}<Rn1!-<{HZ+uI7E3^k26Jg=Ze?VYh-eo}8g+-1Qj};KO95ybXdGx1j1ysUa$$KWX>N2jG%O%7WGo"
                ";<DU@gzO@je-hm=y3Xe?+fXaZ;jXb5NvPKT6zlqhHpXenqKO9E&cXdGx1j1ysUa$$KWX>N2jG%O%8WGo;<DU@gzO"
                "^5+?hm=y3Xc=e%Xa#5^XbVn<lzo&aXcUYFb!l#NHZ+uI7E3^ACy8hmOBQv9lv0#vENCorlr8"
            ),
        },
    ),
    "haplotype_reconstruction.science.assembly.join.proposals._exclusion_scores": (
        {
            "text": ("(Array(float64, 3, 'C', False, aligned=True),)"),
            "pickle": (
                "fCQB`0RR910001t8E$oLVqq?0Z*pZWba`-Pb1q?Gb98cHV|0{^6JK<BaAjX|Wn*t{b98cbV{~tFl#`T<2V-w=d2("
                "fEl#C8va%E$0ZgX^Ubz^jIa+H&lj2UiqZDL_AV{dY0E_8WtWpgfWaCvljaAk9pj0Hh*a$$Ltla!1IVs&Y3bZKsLl"
                "#B*%Vrpe$bd-~nPKT6zlqie^WOR9OWt3<FXa$TIZgp*9VJ>5Da%C=bd2nTOE^}jHY+-V9l#B&NY;R$7l#`Tb3r>f"
                "WeUvDS1a4t%Wt5BuW^8X^bT%}Uj0j?Bba!cFbZC@IK#T=nV{c?-luI0Shm=y3j0A3EX>F8C1B?c2VR>(Lbd-z%Lz"
                "HL~j1^&Wa$$KWW^8X^bT%|BATwkvAVVpXXctYT0d<FzQj~?1E&"
            ),
        },
    ),
    "haplotype_reconstruction.science.assembly.join.proposals._improve_permutation": (
        {
            "text": ("(Array(float64, 2, 'C', False, aligned=True), Array(int64, 1, 'C', False, aligned=True))"),
            "pickle": (
                "fCQDR0RR910001t8E$oLVqq?0Z*pZWba`-Pb1q?Gb98cHV|0{^6JK<BaAjX|Wn*t{b98cbV{~tFl#`T<2V-w=d2("
                "fEl#C8va%E$0ZgX^Ubz^jIa+H&lj2UiqZDL_AV{dY0E_8WtWpgfWaCvljaAk9pj0Hh*a$$Ltla!1IVs&Y3bZKsLl"
                "#B*%Vrpe$bd-~nPKT6zlqie^WOR9OWt3<FXa$TIZgp*9VJ>5Da%C=bd2nTOE^}jHY+-V9l#B&NY;R$7l#`Tb3r>f"
                "WeUvDS1a4t%Wt5BuW^8X^bT%}Uj0j?Bba!cFbZC@IK#T=nV{c?-luI0Shm=y3j0A3EX>F8C0*nT1VR>(Lbd-z%Lz"
                "HL~j1^&Wa$$KWW^8X^bT%|BATneuAVVpXXctX=0d<FzQj}-{Xa#5pXbVn<lzo&aXbxxsXa#5wj0Z_>bY*8{a+H&l"
                "XbVn<lzo&aXcUYEX>N2jG?Zu-OF)bUb7^O8Wn`3yXctQwb%&HvlxP}D0caa&9B34b6Jc_4VR<NNZge&@EFdvtEFe"
                "QElxP=Cg8_Ailv0$2lr8"
            ),
        },
    ),
    "haplotype_reconstruction.science.assembly.join.proposals._possibly_novel": (
        {
            "text": (
                "(Array(float64, 3, 'C', False, aligned=True), Array(int64, 2, 'C', False, aligned=True),"
                " Array(float64, 1, 'C', False, aligned=True))"
            ),
            "pickle": (
                "fCQE40RR910001t8E$oLVqq?0Z*pZWba`-Pb1q?Gb98cHV|0{^6JK<BaAjX|Wn*t{b98cbV{~tFl#`T<2V-w=d2("
                "fEl#C8va%E$0ZgX^Ubz^jIa+H&lj2UiqZDL_AV{dY0E_8WtWpgfWaCvljaAk9pj0Hh*a$$Ltla!1IVs&Y3bZKsLl"
                "#B*%Vrpe$bd-~nPKT6zlqie^WOR9OWt3<FXa$TIZgp*9VJ>5Da%C=bd2nTOE^}jHY+-V9l#B&NY;R$7l#`Tb3r>f"
                "WeUvDS1a4t%Wt5BuW^8X^bT%}Uj0j?Bba!cFbZC@IK#T=nV{c?-luI0Shm=y3j0A3EX>F8C1B?c2VR>(Lbd-z%Lz"
                "HL~j1^&Wa$$KWW^8X^bT%|BATwkvAVVpXXctYT0d<FzQj}-{Xa#5pXbVn<lzo&aXbxxsXa#5wj0Z_>bY*8{a+H&l"
                "XbVn<lzo&aXcUYEX>N2jG?Zu-OF)bUb7^O8Wn`3yXctQwb%&HvlxP}D0%#j(9B34b6Jc_4VR<NNZge&@EFdyuEFe"
                "QElxP=Chyitnlv0#v0%!$j2xtpVhm?JkC}<978E6_y0caa&9B34b6=8C6VR<NKY;R$7HZ&|CF=Q+tLn)MK7fpfzb"
                "%&Hvl!ufq"
            ),
        },
    ),
    "haplotype_reconstruction.science.assembly.join.proposals._stream_static_gains": (
        {
            "text": (
                "(Array(float64, 1, 'C', False, aligned=True), Array(int64, 1, 'C', False, aligned=True),"
                " Array(int64, 1, 'C', False, aligned=True), Array(int64, 2, 'C', False, aligned=True), A"
                "rray(int64, 2, 'C', False, aligned=True), Array(int64, 1, 'C', False, aligned=True), Arr"
                "ay(int64, 1, 'C', False, aligned=True), Array(float64, 1, 'C', False, aligned=True), Arr"
                "ay(float64, 2, 'C', False, aligned=True), int64)"
            ),
            "pickle": (
                "fCQB?0ssI20000ej2UiqZDL_AV{dY0E_8WtWpgfJVsmtIVPkZZj1ymUd2nT4a%E$0ZgX^Ubz^jIa+H&lj0a<HaCv"
                "fNXOxT%Uvgz*Z*FsRa&=>LZ*r8Al#Cf}b!}o{E@N+UWiE7iaAk8YZg6>Yd2nTOl#B&Ia&lpLl#`T<2x4_<Y;<XEb"
                "Ciq*Z(?d?V|0|0lun0~eUvDS1!Q!2aAlNe0%!${7;bfKVqq?0Z*pZWba`-Pb1rjZVQgV?bCiq)Mr?0kbd-~nXbVn"
                "<lzo&aj0A3BZDo{<2WD(<VRSY$l#B>sX>@mKWOQhhOF)bTUt@1%Wt2-Cb%&Hvl#B#!WNB@bO96}qY+-qCb##=B0Y"
                "j8%6pR&Na&lpLC}wPLVRSY$EFdvtEFeQElxP=Cf&q1hlv0#v0%!$j2xtpVhm?JkC}<970%!$j4~z#%Zggd5Wpb30"
                "lxPc1hm?JkC}<Rn1!-<{HZ+uI7E3^k26Jg=Ze?VYh-eo}8g+-1Qj};KO95ybXdGx1j1ysUa$$KWX>N2jG%O%7WGo"
                ";<DU@gzO@je-hm=y3Xe?*~Xa#5pXbVn<lzo&aXbxy8Xc|ibXd7r8XcUYSVRCX|c_?XabT%|BATneuAVVpXXctY00"
                "d<FzQj};hXe?+fXdh?-Xa#5pXbVn<lzo&aXbxx@Xc|ibXd7r8XcUYUVRCX|c_?OVZ((#cG%O%8WGo;<DU@gzO??4"
                "%hm=y3Xeo4*E&"
            ),
        },
    ),
    "haplotype_reconstruction.science.assembly.join.proposals.block_restart_scores": (
        {
            "text": (
                "(Array(float64, 1, 'C', False, aligned=True), Array(int64, 1, 'C', False, aligned=True),"
                " Array(int64, 1, 'C', False, aligned=True), Array(int64, 1, 'C', False, aligned=True), A"
                "rray(int64, 2, 'C', False, aligned=True), float64, int64)"
            ),
            "pickle": (
                "fCQEB0RR910000ej2UiqZDL_AV{dY0E_8WtWpgfJVsmtIVPkZZj1ymUd2nT4a%E$0ZgX^Ubz^jIa+H&lj0a<HaCv"
                "fNXOxT%Uvgz*Z*FsRa&=>LZ*r8Al#Cf}b!}o{E@N+UWiE7iaAk8YZg6>Yd2nTOl#B&Ia&lpLl#`T<2x4_<Y;<XEb"
                "Ciq*Z(?d?V|0|0lun0~eUvDS1!Q!2aAlNe0%!${7;bfKVqq?0Z*pZWba`-Pb1rjZVQgV?bCiq)Mr?0kbd-~nXbVn"
                "<lzo&aj0A3BZDo{<2WD(<VRSY$l#B>sX>@mKWOQhhOF)bTUt@1%Wt2-Cb%&Hvl#B#!WNB@bO96}qY+-qCb##=B0Y"
                "j8%6pR&Na&lpLC}wPLVRSY$EFdvtEFeQElxP=Cf&q1hlv0#v0%!$j2xtpVhm?JkC}<970%!$j4~z#%Zggd5Wpb30"
                "lxPc1hm?JkC}<Rn1!-<{HZ+uI7E3^k26Jg=Ze?VYh-eo}8g+-1Qj};KO95ybXdGx1j1ysUa$$KWX>N2jG%O%7WGo"
                ";<DU@gzO@je-hm=y3Xe?+fXaZ;jXb5NvPKT6zlqhHpXenqKO9E&cXdGx1j1ysUa$$KWX>N2jG%O%8WGo;<DU@gzO"
                "^5+?hm=y3Xc=fJbd)X"
            ),
        },
    ),
    "haplotype_reconstruction.science.assembly.join.proposals.conditional_fields": (
        {
            "text": (
                "(Array(float64, 3, 'C', False, aligned=True), Array(int32, 2, 'C', False, aligned=True),"
                " Array(int64, 1, 'A', False, aligned=True), int64, int64)"
            ),
            "pickle": (
                "fCQBk0ssI20000ej2UiqZDL_AV{dY0E_8WtWpgfJVsmtIVPkZZj1ymUd2nT4a%E$0ZgX^Ubz^jIa+H&lj0a<HaCv"
                "fNXOxT%Uvgz*Z*FsRa&=>LZ*r8Al#Cf}b!}o{E@N+UWiE7iaAk8YZg6>Yd2nTOl#B&Ia&lpLl#`T<2x4_<Y;<XEb"
                "Ciq*Z(?d?V|0|0lun0~eUvDS1!Q!2aAlNe0%!${7;bfKVqq?0Z*pZWba`-Pb1rjZVQgV?bCiq)Mr?0kbd-~nXbVn"
                "<lzo&aj0A3BZDo{<2WD(<VRSY$l#B>sX>@mKWOQhhOF)bTUt@1%Wt2-Cb%&Hvl#B#!WNB@bO9PAsY+-qCb##=B0Y"
                "j8%6pR&Na&lpLC}wPLVRSY$EFd#vEFeQElxP=Cr2%z^lv0#v0%!$j2xtpVhm?JkC}<970%!$j4~z#%Zggd5Wpb30"
                "lxPc1hm?JkC}<Rn1!-<{GcuHD7E2(E26Jg=Ze?VYh-eo}8Fh!0Qj};KO9E&cXdGx1j1ysUa$$KWX>N2gGAtl6WGo"
                ";<DU@gzO_u?6hm=y3XaZ;jXb5NvPKT6zlqhHpXaZ;jXd`F~PKT6zlqhHvj0I_KbT%}UXckLAXeWqh7fTv-hm=y3X"
                "c|iaXd8?HL6m3|j1ysUa$$KWX>N2jG%O%7WGo;-DU@gzP2~Y~hm=y3XftRtbd)X"
            ),
        },
    ),
    "haplotype_reconstruction.science.assembly.join.proposals.fixed_paint_delta": (
        {
            "text": (
                "(Array(float64, 3, 'C', False, aligned=True), Array(int32, 2, 'A', False, aligned=True),"
                " Array(int64, 1, 'A', False, aligned=True), Array(int64, 1, 'A', False, aligned=True), i"
                "nt64)"
            ),
            "pickle": (
                "fCQBk0ssI20000ej2UiqZDL_AV{dY0E_8WtWpgfJVsmtIVPkZZj1ymUd2nT4a%E$0ZgX^Ubz^jIa+H&lj0a<HaCv"
                "fNXOxT%Uvgz*Z*FsRa&=>LZ*r8Al#Cf}b!}o{E@N+UWiE7iaAk8YZg6>Yd2nTOl#B&Ia&lpLl#`T<2x4_<Y;<XEb"
                "Ciq*Z(?d?V|0|0lun0~eUvDS1!Q!2aAlNe0%!${7;bfKVqq?0Z*pZWba`-Pb1rjZVQgV?bCiq)Mr?0kbd-~nXbVn"
                "<lzo&aj0A3BZDo{<2WD(<VRSY$l#B>sX>@mKWOQhhOF)bTUt@1%Wt2-Cb%&Hvl#B#!WNB@bO9PAsY+-qCb##=B0Y"
                "j8%6pR&Na&lpLC}wPLVRSY$EFd#vEFeQElxP=C#sPJQlv0#v0%!$j2xtpVhm?JkC}<970%!$j4~z#%Zggd5Wpb30"
                "lxPc1hm?JkC}<Rn1!-<{GcuHD7E2(E26Jg=Ze?VYh-eo}8Fh!0Qj};KO9E&ci~&KEXcUYSVRCX|c_?XabTcw6ATn"
                "euAVDdVXctY_0d<FzQj}-{Xa#5pXbVn<lzo&aXbxxsXa#5^XbVn<lzo&aXcUYEX>N2jG?Zu-OF(ERh-eo}8g+-1Q"
                "j};KO95ybXewwFj1ysUa$$KWX>N2jG%O%7WGo;-DU@gzP0s;!hm=y3Xg6pybd)X"
            ),
        },
    ),
    "haplotype_reconstruction.science.assembly.join.proposals.prefix_scores": (
        {
            "text": (
                "(Array(float64, 1, 'C', False, aligned=True), Array(int64, 1, 'C', False, aligned=True),"
                " Array(int64, 1, 'C', False, aligned=True), Array(int64, 1, 'C', False, aligned=True), A"
                "rray(int64, 2, 'C', False, aligned=True), Array(int64, 2, 'C', False, aligned=True), Arr"
                "ay(int64, 1, 'C', False, aligned=True), Array(int64, 1, 'C', False, aligned=True), float"
                "64, int64)"
            ),
            "pickle": (
                "fCQEH0RR910000ej2UiqZDL_AV{dY0E_8WtWpgfJVsmtIVPkZZj1ymUd2nT4a%E$0ZgX^Ubz^jIa+H&lj0a<HaCv"
                "fNXOxT%Uvgz*Z*FsRa&=>LZ*r8Al#Cf}b!}o{E@N+UWiE7iaAk8YZg6>Yd2nTOl#B&Ia&lpLl#`T<2x4_<Y;<XEb"
                "Ciq*Z(?d?V|0|0lun0~eUvDS1!Q!2aAlNe0%!${7;bfKVqq?0Z*pZWba`-Pb1rjZVQgV?bCiq)Mr?0kbd-~nXbVn"
                "<lzo&aj0A3BZDo{<2WD(<VRSY$l#B>sX>@mKWOQhhOF)bTUt@1%Wt2-Cb%&Hvl#B#!WNB@bO96}qY+-qCb##=B0Y"
                "j8%6pR&Na&lpLC}wPLVRSY$EFdvtEFeQElxP=Cf&q1hlv0#v0%!$j2xtpVhm?JkC}<970%!$j4~z#%Zggd5Wpb30"
                "lxPc1hm?JkC}<Rn1!-<{HZ+uI7E3^k26Jg=Ze?VYh-eo}8g+-1Qj};KO95ybXdGx1j1ysUa$$KWX>N2jG%O%7WGo"
                ";<DU@gzO@je-hm=y3Xe?+fXaZ;jXb5NvPKT6zlqhHpXenqKO9E&cXdGx1j1ysUa$$KWX>N2jG%O%8WGo;<DU@gzO"
                "^5+?hm=y3XfbFkXe?+MXeo4*E&"
            ),
        },
    ),
    "haplotype_reconstruction.science.assembly.join.splice._junction_scores": (
        {
            "text": (
                "(Array(float64, 1, 'C', False, aligned=True), Array(float64, 1, 'C', False, aligned=True"
                "), Array(int64, 1, 'C', False, aligned=True), Array(int64, 1, 'C', False, aligned=True),"
                " Array(int64, 2, 'C', False, aligned=True), Array(float64, 1, 'C', False, aligned=True))"
            ),
            "pickle": (
                "fCQE90RR910000ej2UiqZDL_AV{dY0E_8WtWpgfJVsmtIVPkZZj1ymUd2nT4a%E$0ZgX^Ubz^jIa+H&lj0a<HaCv"
                "fNXOxT%Uvgz*Z*FsRa&=>LZ*r8Al#Cf}b!}o{E@N+UWiE7iaAk8YZg6>Yd2nTOl#B&Ia&lpLl#`T<2x4_<Y;<XEb"
                "Ciq*Z(?d?V|0|0lun0~eUvDS1!Q!2aAlNe0%!${7;bfKVqq?0Z*pZWba`-Pb1rjZVQgV?bCiq)Mr?0kbd-~nXbVn"
                "<lzo&aj0A3BZDo{<2WD(<VRSY$l#B>sX>@mKWOQhhOF)bTUt@1%Wt2-Cb%&Hvl#B#!WNB@bO96}qY+-qCb##=B0Y"
                "j8%6pR&Na&lpLC}wPLVRSY$EFdvtEFeQElxP=Cf&q1hlv0#vA7}z-1!xFp3r>fWeUvC@4rl^s1!xb92T5*pWoKn_"
                "l#`Tb3r>fWeUvC@6pRIFZge&@lxP-9K#T@+X=iR_WR!?#7fTv-hm=y3Xc|iaXd7r8XcUYSVRCX|c_?XabT%|BATe"
                "YtAVVpXXctX`0d<FzQj};cXaZ;jXb5NvPKT6zlqhHpXenqKO9E&cXdGx1j1ysUa$$KWX>N2jG%O%8WGo;<DU@gzO"
                "^5+?hm=y3XdiTxE&"
            ),
        },
    ),
    "haplotype_reconstruction.science.assembly.join.splice.score_splices": (
        {
            "text": (
                "(Array(float64, 1, 'C', False, aligned=True), Array(int64, 1, 'C', False, aligned=True),"
                " Array(int64, 1, 'C', False, aligned=True), Array(int64, 1, 'C', False, aligned=True), A"
                "rray(int64, 2, 'C', False, aligned=True), float64, int64)"
            ),
            "pickle": (
                "fCQEB0RR910000ej2UiqZDL_AV{dY0E_8WtWpgfJVsmtIVPkZZj1ymUd2nT4a%E$0ZgX^Ubz^jIa+H&lj0a<HaCv"
                "fNXOxT%Uvgz*Z*FsRa&=>LZ*r8Al#Cf}b!}o{E@N+UWiE7iaAk8YZg6>Yd2nTOl#B&Ia&lpLl#`T<2x4_<Y;<XEb"
                "Ciq*Z(?d?V|0|0lun0~eUvDS1!Q!2aAlNe0%!${7;bfKVqq?0Z*pZWba`-Pb1rjZVQgV?bCiq)Mr?0kbd-~nXbVn"
                "<lzo&aj0A3BZDo{<2WD(<VRSY$l#B>sX>@mKWOQhhOF)bTUt@1%Wt2-Cb%&Hvl#B#!WNB@bO96}qY+-qCb##=B0Y"
                "j8%6pR&Na&lpLC}wPLVRSY$EFdvtEFeQElxP=Cf&q1hlv0#v0%!$j2xtpVhm?JkC}<970%!$j4~z#%Zggd5Wpb30"
                "lxPc1hm?JkC}<Rn1!-<{HZ+uI7E3^k26Jg=Ze?VYh-eo}8g+-1Qj};KO95ybXdGx1j1ysUa$$KWX>N2jG%O%7WGo"
                ";<DU@gzO@je-hm=y3Xe?+fXaZ;jXb5NvPKT6zlqhHpXenqKO9E&cXdGx1j1ysUa$$KWX>N2jG%O%8WGo;<DU@gzO"
                "^5+?hm=y3Xc=fJbd)X"
            ),
        },
    ),
    "haplotype_reconstruction.science.assembly.pairs.emission_pair_index": (
        {
            "text": ("(int64, int64)"),
            "pickle": (
                "fCQD!000000001t8E$oLVqq?0Z*pZWba`-Pb1q?Gb98cHV|0{^6JK<BaAjX|Wn*t{b98cbV{~tFl#`T<2V-w=d2("
                "fEl#C8va%E$0ZgX^Ubz^jIa+H&lj2LcpZDL_AV{dY0E_8WtWpgfbV_|Gza&wf72T5*pWoKn_l#`T<2x4_<Y;<XEb"
                "Ciq*Z(?d?V|0|0lun0~eUvDS1a4t%Wt5BsX>N2jG?a`8Vrg`DX=HS0luJO226Jg=Ze?VYh>QhaV{c?-luH_Qhm=y"
                "3XcUH&E&"
            ),
        },
    ),
    "haplotype_reconstruction.science.assembly.pairs.splice_column": (
        {
            "text": ("(int64, int64, int64)"),
            "pickle": (
                "fCQD$000000001t8E$oLVqq?0Z*pZWba`-Pb1q?Gb98cHV|0{^6JK<BaAjX|Wn*t{b98cbV{~tFl#`T<2V-w=d2("
                "fEl#C8va%E$0ZgX^Ubz^jIa+H&lj2LcpZDL_AV{dY0E_8WtWpgfbV_|Gza&wf72T5*pWoKn_l#`T<2x4_<Y;<XEb"
                "Ciq*Z(?d?V|0|0lun0~eUvDS1a4t%Wt5BsX>N2jG?a`8Vrg`DX=HS0luJO226Jg=Ze?VYh>QhaV{c?-luH_Qhm=y"
                "3XcTA^hm<Y"
            ),
        },
    ),
    "haplotype_reconstruction.science.assembly.pairs.state_pairs": (
        {
            "text": ("(int64, Array(bool, 1, 'C', False, aligned=True))"),
            "pickle": (
                "fCQCk0RR910001t8E$oLVqq?0Z*pZWba`-Pb1q?Gb98cHV|0{^6JK<BaAjX|Wn*t{b98cbV{~tFl#`T<2V-w=d2("
                "fEl#C8va%E$0ZgX^Ubz^jIa+H&lj2LcpZDL_AV{dY0E_8WtWpgfbV_|Gza&wf72T5*pWoKn_l#`T<2x4_<Y;<XEb"
                "Ciq*Z(?d?V|0|0lun0~eUvDS1a4t%Wt5BsX>N2jG?a`8Vrg`DX=HS0luJO226Jg=Ze?VYh>QhaV{c?-luH_Qhm=y"
                "3XaZ;jj2UiqZDL_AV{dY0E_8WtWpgfWaCvljaAk9pj0Hh*a$$Ltlay!+PKT6zlqie^WOR9OWt3<FXa#5nj0Zw*Z)"
                "{~@Zj_UhXbVn<lzo&aXby}7VsCG3lxPx55_N}^Qk0AYZe(d~luH4O25ez@Z*_E(i~&QGXby}LVRCX|c_?CUZ)_|e"
                "F=Q+tLn)MK5>2TAb%&Hvl!lZp"
            ),
        },
        {
            "text": ("(int64, none)"),
            "pickle": (
                "fCQBf0RR910001t8E$oLVqq?0Z*pZWba`-Pb1q?Gb98cHV|0{^6JK<BaAjX|Wn*t{b98cbV{~tFl#`T<2V-w=d2("
                "fEl#C8va%E$0ZgX^Ubz^jIa+H&lj2LcpZDL_AV{dY0E_8WtWpgfbV_|Gza&wf72T5*pWoKn_l#`T<2x4_<Y;<XEb"
                "Ciq*Z(?d?V|0|0lun0~eUvDS1a4t%Wt5BsX>N2jG?a`8Vrg`DX=HS0luJO226Jg=Ze?VYh>QhaV{c?-luH_Qhm=y"
                "3XaZ;jj1_KmZDL_AV{dY0E_8WtWpgfVX>((gj0jF|Ze>(?aAlN}lxPc1hm?JkC}<9h1a5C`Wt3<VO9OR>lv0$2lr"
                "8"
            ),
        },
        {
            "text": ("(int64, omitted(default=None))"),
            "pickle": (
                "fCQB|0RR910001t8E$oLVqq?0Z*pZWba`-Pb1q?Gb98cHV|0{^6JK<BaAjX|Wn*t{b98cbV{~tFl#`T<2V-w=d2("
                "fEl#C8va%E$0ZgX^Ubz^jIa+H&lj2LcpZDL_AV{dY0E_8WtWpgfbV_|Gza&wf72T5*pWoKn_l#`T<2x4_<Y;<XEb"
                "Ciq*Z(?d?V|0|0lun0~eUvDS1a4t%Wt5BsX>N2jG?a`8Vrg`DX=HS0luJO226Jg=Ze?VYh>QhaV{c?-luH_Qhm=y"
                "3XaZ;jj1_KmZDL_AV{dY0E_8WtWpgfVX>((gj0aC`X>@dDWR#PXXbVn<lzo&aj0RtJVQh6}lunEaUv^<^b!A^`Wq"
                "FiNXby}OZ*6IGbY)~HWMyVyb!>D!PH%2yDU@guO@0A&hm=y3hLkP"
            ),
        },
    ),
}


# Saved depth-2 chr3 tail: noncontiguous shared GL and immutable local row codes.
# This is an additional supported storage signature, not a numerical route.
SIGNATURES["haplotype_reconstruction.science.assembly.join.emissions._binned_log_emissions"] += (
    {
        "text": (
            "(Array(float64, 1, 'C', False, aligned=True), int64, Array(float32, 3, 'A', True, aligned=True),"
            " int64, Array(int8, 2, 'C', True, aligned=True), Array(int32, 2, 'C', False, aligned=True), Arra"
            "y(bool, 1, 'C', False, aligned=True), Array(bool, 1, 'C', False, aligned=True), int64)"
        ),
        "pickle": (
            "fCQB!0{{R30000ej2UiqZDL_AV{dY0E_8WtWpgfJVsmtIVPkZZj1ymUd2nT4a%E$0ZgX^Ubz^jIa+H&lj0a<HaCvfNXOxT%U"
            "vgz*Z*FsRa&=>LZ*r8Al#Cf}b!}o{E@N+UWiE7iaAk8YZg6>Yd2nTOl#B&Ia&lpLl#`T<2x4_<Y;<XEbCiq*Z(?d?V|0|0lu"
            "n0~eUvDS1!Q!2aAlNe0%!${7;bfKVqq?0Z*pZWba`-Pb1rjZVQgV?bCiq)Mr?0kbd-~nXbVn<lzo&aj0A3BZDo{<2WD(<VRS"
            "Y$l#B>sX>@mKWOQhhOF)bTUt@1%Wt2-Cb%&Hvl#B#!WNB@bO96}qY+-qCb##=B0Yj8%6pR&Na&lpLC}wPLVRSY$EFdvtEFeQ"
            "ElxP=CgaLJjlv0#v0%!$j4~z#%Zggd5Wpb30lxPc1hm?JkC}<Rn1!-<{HZ+uI7E3^k26Jg=Ze?VYh-eo}8g+-1Qj}-{Xa#5p"
            "XbVn<lzo&aj0bIXbYWs_Wt53%4rl^s1!xgy3r>fWeUvC@6pROEY;R$7GcuHD7E2&#7fTy;hm=y3Xc|icXd8?HL6m3|j2?1jV"
            "PtP^Y<VDIa&lpLC}wPLVRSPxEFd#vEFeKClxP=CVFGoBlv0#vCujm_1!xFp3r>fWeUvC@Dv4+gXaZ;jXd!3|PKT6zlqhHvj0"
            "9<JbU2h~7E1_dC5UJjOBZ#Alv0#v8cPCb8)zJ86pR~kWnpA*ZftoVVRCX|c_?XabT}*^GGr_uLn)MK7foyeb%&HvlxPBI1!x"
            "Fp3r>fWeUvC@4rl^s1!y5?3r>fWeUvC@6pRIFZgev;lxP-9AZR6sXctQvb%&HvlxP}D0%#j(9B34b6Jc_4VR<NNZgev;EFdy"
            "uEFeQElxP=CYXWtLlv0#v0%!$j2xtpVhm?JkC}<970%!$j4~z#wZ*Od6VQ!R@lxPc1hm?JkC}<Rn1Y&P*Y?NphOA>X5lv0#v"
            "8cP9a8)zJ86pRvKa&lpLC}MAKY%Cx#WGo;<DU@gzO^g9`hm=y3Xj5n>bd)X"
        ),
    },
)


def overloads_for(module: str, name: str) -> tuple:
    """Return exact argument tuples for a canonical science dispatcher."""
    canonical = module.replace("haplotype_reconstruction.assembly.", "haplotype_reconstruction.science.assembly.")
    return tuple(decode_signature(item) for item in SIGNATURES.get(canonical + "." + name, ()))


__all__ = ["SIGNATURES", "overloads_for"]
