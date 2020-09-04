# Projet JADIS

Le projet JADIS est issu d'une collaboration scientifique entre la Bibliothèque nationale de France (BnF) et l'Ecole Polytechnique Fédérale de Lausanne (EPFL). Les buts du projet sont les suivants:

* Développer un algorithme permettant de géolocaliser et réaligner automatiquement les collections cartographiques avec une précision au niveau de la rue.
* Réaligner les résultats sur la base de données des noms de rues historiques pour permettre la fouille des cartes de Paris par les noms de rues d’époque.

L'algorithme comprend deux axes principaux. Le premier vise à vectoriser automatiquement les cartes à l'aide d'un réseau neuronal. Le second vise à créer un réseau de similarité pour réaligner les cartes entre elles. Le tronc principal du programme permet ensuite d'identifier les similarités géométriques entre les cartes vectorisées pour déterminer la position des cartes les unes par rapport aux autres et également par rapport au tissu urbain actuel. 

## [Documentation](https://github.com/BnF-jadis/projet/blob/master/Jadis_manuel.pdf)

Un manuel d'utilisation très détaillé est disponible [sur ce lien](https://github.com/BnF-jadis/projet/blob/master/Jadis_manuel.pdf). Le programme est conçu pour être entièrement utilisable par des utilisateurs qui ne savent pas programmer.

La documentation concernant le fonctionnement des algorithmes de segmentation et de réalignement est disponible sur ce lien.

## Installation facile

1. Installez [Anaconda](https://docs.anaconda.com/anaconda/install/)
2. [Téléchargez le programme JADIS](https://github.com/BnF-jadis/projet/archive/master.zip), décompressez-le et placez-
le dans le dossier de votre choix, par exemple sur le Bureau
3. [Téléchargez le réseau de neurones entraîné](https://drive.google.com/file/d/13iRsEwFv9tTe68v5d_dXlEAJj9sn0qsb/view?usp=sharing),
décompressez-le et placez-le dans le dossier JADIS
4. Ouvrez une invite de commande. Sur Windows 7, 8 ou 10, vous pourrez utiliser l’application Invite
de commandes, installée par défaut et accessible via la recherche. À défaut, vous pouvez utiliser par exemple Anaconda prompt. Sur Linux/Fedora ou sur Mac, utilisez Terminal.
5. Vous vous trouvez dans l’un des dossiers de votre ordinateur, en général le dossier source de votre compte. Le nom du dossier est indiqué à gauche de votre curseur, par exemple
```Users\Remi``` ou ```~Remi $```. Sur Unix (Mac, Linux), vous pouvez taper  
``` ls ```
Pour lister les fichiers situés dans le répertoire où vous vous trouvez.
6. Naviguez jusqu’au dossier JADIS dans votre ordinateur. Par exemple, s’il se trouve sur votre
Bureau, le chemin sera probablement Desktop/JADIS :
``` cd Desktop/JADIS ```
7. Utilisez la fonction setup.py pour installer le programme :
``` python setup.py ```
8. Lorsque l’invite de commande vous demande si vous souhaitez continuer, tapez y puis retour.
9. Si l’installation échoue, vérifiez votre connexion internet et ré-essayez une deuxième fois.

## Licence
CC BY 3.0 FR (Résumé ci-dessous)

Vous êtes autorisé à :
* Partager — copier, distribuer et communiquer le matériel par tous moyens et sous tous formats
* Adapter — remixer, transformer et créer à partir du matériel pour toute utilisation, y compris commerciale.

Cette licence est acceptable pour des œuvres culturelles libres.
* L'Offrant ne peut retirer les autorisations concédées par la licence tant que vous appliquez les termes de cette licence.

Selon les conditions suivantes :
* Attribution — Vous devez créditer l'Œuvre, intégrer un lien vers la licence et indiquer si des modifications ont été effectuées à l'Oeuvre. Vous devez indiquer ces informations par tous les moyens raisonnables, sans toutefois suggérer que l'Offrant vous soutient ou soutient la façon dont vous avez utilisé son Oeuvre.
* Pas de restrictions complémentaires — Vous n'êtes pas autorisé à appliquer des conditions légales ou des mesures techniques qui restreindraient légalement autrui à utiliser l'Oeuvre dans les conditions décrites par la licence.

