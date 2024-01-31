# Code réalisé par Nathan Bizon
# Pour lancer le programme :
# - Installer streamlit : pip install streamlit
# - ouvrez l'invite de commande et exécutez : cd chemin/vers/le/fichier
# - tappez la commande : streamlit run main.py

import streamlit as st  # C'est une bibliothèque permettant de créer une interface utilisateur web
# (documentation : https://docs.streamlit.io/)
import pandas as pd  # Permet une utilisation simple des tableaux de données
import math
import numpy as np
from scipy.integrate import odeint  # équa difs

g = 9.80665  # constante gravitationnelle

voitures = [
    "Modèle 1 : Dodge Charger R/T, 1970",
    "Modèle 2 : Toyota Supra Mark IV, 1994",
    "Modèle 3 : Chevrolet Yenko Camaro 1969",
    "Modèle 4 : Mazda RX-7 FD",
    "Modèle 5 : Nissan Skyline GTR-R34, 1999",
    "Modèle 6 : Mitsubishi Lancer Evolution VII"
]

circuit = [
    "Pente",
    "Looping",
]

sel_voiture = lambda n: voitures[n]  # fonction qui permet de donner un nom pou la séléction des voitures
sel_circuit = lambda n: circuit[n]  # même chose pour la séléction du booster

data = pd.read_csv("voitures.csv")  # charge les données des voitures

st.set_page_config(page_title="Fast And Furious !")
st.markdown("# Fast And Furious ! Simulation des voitures sur le circuit")  # écris sur le site avec la syntaxe markdown
st.write("Spécifications des voitures")  # écris sur le site
st.dataframe(data, hide_index=True)  # Affiche le tableau
modele = st.selectbox("Séléction de la voiture", list(range(6)), format_func=sel_voiture)  # la boite de séléction
turbo = st.checkbox('Ajouter le turbo')  # ajouter la séléction des ailerons et du turbo
if turbo:
    turbo_circuit = st.radio("Partie du circuit où appliquer le turbo", list(range(2)), format_func=sel_circuit)
else:
    turbo_circuit = None

ailerons = st.checkbox('Ajouter les ailerons et la jupe avant')

frottements = st.toggle('Mode frottement')
def avance(pb, n):  # fonction pour la barre de chargement
    n += 12
    pb.progress(n, text="calcul en cours...")
    return n

def looping(masse,g,cx,mv_air,sx,acc,vfp):
    l = 6  # rayon du looping
    v = vfp  # vitesse initiale
    frot = 0.1
    theta_dot_init = v / l  # vitesse angulaire initiale (theta_point(0))
    # SANS FROTTEMENTS
    def theta(y, t, m, g, r, F):  # équation différentielle de theta
        try:
            # y[0] est la position (y), y[1] est la vitesse (u = y')
            dydt = [y[1], (-m * g * np.sin(y[0])+ F) / (m * r)]
        except OverflowError:  # la vitesse peut descendre très rapidement et peut atteindre des valeurs négatives
            dydt = y  # énormes, on ajoute donc une sortie au cas où la valeur n'est pas supporté
        return dydt

    # AVEC FROTTEMENTS
    def theta_f(y, t, m, g, r, Cx, rho, Sx, F):  # équation différentielle de theta
        try:
            # y[0] est la position (y), y[1] est la vitesse (u = y')
            dydt = [y[1], (-m * g * np.sin(y[0]) - 0.5 * Cx * rho * Sx * (r * y[1] ** 2) - frot * (
                    r * (y[1] ** 2) )+ F) / (m * r)]
        except OverflowError:  # la vitesse peut descendre très rapidement et peut atteindre des valeurs négatives
            dydt = y  # énormes, on ajoute donc une sortie au cas où la valeur n'est pas supporté
        return dydt

    t = np.linspace(0, 5, 10000)

    F = acc*1.3 if turbo_circuit == 1 else acc

    y0 = [0, theta_dot_init]  # système initial (theta(0) = 0, theta_point(0) = vitesse_angulaire_initiale)
    if frottements:
        solution = odeint(theta_f, y0, t, args=(masse, g, 6, cx, mv_air, sx, F))  # résolution avec odeint
    else:
        solution = odeint(theta, y0, t, args=(masse, g, 6, F))

    def fil(n):  # fonction de filtre pour enlever les valeurs en trop (on ne fait qu'un seul tour de looping)
        return n < 2 * np.pi

    pos = solution[:, 0]  # les positions angulaires de la voiture
    pos = list(filter(fil, pos))  # on enlève les valeurs en trop
    vct = solution[:len(pos) - 1, 1] * 6  # on enlève aussi les valeurs de la vitesse en trop et on multiplie par le
                                          # rayon du looping pour convertir la vitesse angulaire en vitesse
    return vct,t,pos

def ravin_f(vsl,cx,mv_air,sx,cz,sz,g,masse):
    tvx = []  # trajectoire de la voiture sur x
    tvy = []  # trajectoire de la voiture sur y

    vx = vsl
    vy = 0

    tx = 0
    ty = 1

    kx = 0.5 * cx * mv_air * sx
    ky = 0.5 * cz * mv_air * sz

    if ailerons:
        ky *= 1.1

    ay = -g

    dt = 0.0001

    temps_ravin = 0
    while ty > 0:
        try:
            tvx.append(tx)
            tvy.append(ty)
            vx -= (kx / masse) * (vx ** 2) * dt
            tx += vx * dt
            vy += (ay - (ky / masse) * (vy ** 2)) * dt
            ty += vy * dt
            temps_ravin += dt
        except OverflowError:
            break
    return tvx,tvy,temps_ravin,vx

def calculer(m, ailerons, tc):  # m = modele | t = turbo | a = ailerons | tc = turbo_circuit
    n = 0
    prog_bar = st.progress(n, text="Calcul en cours...")
    valeurs = data.to_dict()
    # calculs généraux et accès aux valeurs
    masse = valeurs["Masse (kg)"][m]
    acc = valeurs["Accélération moyenne (m/s²)"][m]
    long = valeurs["Longueur (m)"][m]
    larg = valeurs["Largeur (m)"][m]
    haut = valeurs["Hauteur (m)"][m]
    sx = larg * haut
    sz = long * larg
    cx = valeurs["Cx"][m]
    cz = valeurs["Cz"][m]
    frot = 0.1
    mv_air = 1.225

    if ailerons:  # si l'option des ailerons est activé
        masse += 45
        sz += 0.8
        frot *= 0.95

    n = avance(prog_bar, n)
    # calcul de la vitesse en fin de pente

    # SANS FROTTEMENT
    if not frottements:
        if tc == 0:
            acc_pente = acc * 1.3
        else:
            acc_pente = acc
        a = 0.5 * (g * (2 / 31) + acc_pente)
        c = -31  # opposé de la distance
        delta = -4 * a * c
        # distance positive donc racine positive
        temps_pente = math.sqrt(delta) / (2 * a)
        vfp = a * 2 * temps_pente

    # AVEC FROTTEMENT
    else:
        temps_pente = 0
        v = 0  # vitesse
        d = 0  # position
        acc_pente = acc * 1.3 if tc == 0 else acc
        a = g * (2 / 31) + acc_pente - frot * acc
        k = 0.5 * cx * mv_air * sx
        dt = 0.00001
        while d < 31:
            v += (a - (k / masse) * (v ** 2)) * dt  # a += 1 <=> a = a + 1
            d += v * dt
            temps_pente += dt
        vfp = v  # vfp : vitesse en fin de pente

    n = avance(prog_bar, n)
    # calcul looping : graphique vitesse au cours du temps

    vct,t,pos = looping(masse,g,cx,mv_air,sx,acc,vfp)

    intervale = [0,20]

    vml = 0

    while intervale[1] != round(intervale[0]+0.01,2):
        print(intervale, vml)
        vml = round(sum(intervale)/2,2)
        sim,_,_ = looping(masse,g,cx,mv_air,sx,acc,vml)
        for i in sim:
            if i < 0:
                intervale[0] = vml
                break
        else:
            intervale[1] = vml

    t = t[:len(pos) - 1] # on prend le temps pour l'axe des abscisses de notre graphique
    temps_looping = t[-1]
    gv = pd.DataFrame(np.array([t, vct]).transpose(), columns=['x', 'y']) # explication de transposée sur compte rendu

    n = avance(prog_bar, n)
    # calcul looping : vitesse sortie looping
    vsl = vct[-1] # vsl = vitesse sortie looping
    n = avance(prog_bar, n)
    # calcul ravin : trajectoire de la voiture dans le ravin

    # SANS FROTTEMENT
    if not frottements:
        vr = vsl
        t = np.linspace(0,9/vr,500)
        tvx = vr*t
        tvy = -0.5*g*t**2+1
        temps_ravin = 9/vr
        poly = np.polynomial.Polynomial((81 * g, 0, -2))
        vmr = poly.roots()[1]
        vitesse_fin_ravin = vr


    # AVEC FROTTEMENT
    else:
        tvx,tvy,temps_ravin,vitesse_fin_ravin = ravin_f(vsl, cx, mv_air, sx, cz, sz, g, masse)
        intervale = [0, 20]

        vmr = 0

        while intervale != [vmr, round(vmr + 0.01,2)]:
            vmr = round(sum(intervale) / 2, 2)
            sim,_,_,_ = ravin_f(vmr, cx, mv_air, sx, cz, sz, g, masse)
            if sim[-1] < 9:
                intervale[0] = vmr
            else:
                intervale[1] = vmr

    tv = pd.DataFrame(np.array([tvx, tvy]).transpose(), columns=["x", "y"])
    n = avance(prog_bar, n)
    # calcul fin de la piste (10m de ligne droite)
    if vsl-vmr >= 0:
        if frottements:
            temps_fin_piste = 0
            v = vitesse_fin_ravin  # vitesse
            d = tvx[-1]-9  # position
            a = acc - frot * acc
            k = 0.5 * cx * mv_air * sx
            dt = 0.00001
            while d < 10:
                v += (a - (k / masse) * (v ** 2)) * dt  # a += 1 <=> a = a + 1
                d += v * dt
                temps_fin_piste += dt
        else:
            a = 0.5 *acc
            c = tvx[-1]-19  # opposé de la distance
            poly = np.polynomial.Polynomial((c, 0, a))
            temps_fin_piste = poly.roots()[1]
        tt = temps_pente + temps_looping + temps_ravin + temps_fin_piste
    else:
        tt = -1

    prog_bar.progress(100, text="Calcul terminé !")

    # AFFICHAGE DU COMPTE RENDU

    st.title("La pente")
    "Pente sans frottements"
    st.image("img/pente.png")
    "Pente avec frottements"
    st.image("img/pente_f.png")
    st.metric("Vitesse à la fin de la pente", f"{round(vfp, 2)} m/s")
    st.title("Le looping")
    "Looping sans frottements"
    st.image("img/looping.png")
    "Looping avec frottements"
    st.image("img/looping_f.png")
    st.write("Vitesse de la voiture dans le looping en fonction du temps")
    st.line_chart(gv, x='x', y='y')
    st.metric("Vitesse minimale pour rentrer dans le looping sans tomber", f"{round(vml,2)} m/s")
    st.metric("Vitesse à la sortie du looping", f"{round(vsl,2)} m/s")
    st.title("Le ravin")
    "Ravin sans frottements"
    st.image("img/ravin.png")
    "Ravin avec frottements"
    st.image("img/ravin_f.png")
    st.write("Trajectoire de la voiture dans le ravin")
    st.line_chart(tv, x='x', y='y')
    st.metric("Vitesse minimale pour franchir le ravin", f"{round(vmr,2)} m/s")
    st.title("La fin de la piste")
    "Sans frottements"
    st.image("img/fin_piste.png")
    "Avec frottements"
    st.image("img/fin_piste_f.png")
    st.metric("Temps total pour parcourir le circuit", f"{round(tt,2)} secondes")
    looping_possible = "Oui" if vfp - vml >= 0 else "Non"
    st.metric("Le looping est possible", looping_possible, delta=round(vfp - vml,2))
    ravin_possible = "Oui" if vsl - vmr >= 0 else "Non"
    st.metric("Le ravin est possible", ravin_possible, delta=round(vsl - vmr,2))
    st.divider()

st.button("Calculer", on_click=calculer, args=(modele, ailerons, turbo_circuit))
