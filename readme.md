# Le projet *Pylearn*

est un projet de **machine learning** par lequel j'envisage avant tout de booster mes connaissances dans le domaine de l'intelligence rtificielle mais aussi d'expérimenter certaines idées et de les mettre à disposition de qui le voudrait.  
Ce repo comporte le code source de ayant permis de construire à la fois l'IA et l'architecture du site du projet où vous pourrez tester l'IA.  
Si vous souaitez exploiter ce projet vous pouvez soit:
- Consulter le site du projt via le lien: [pylearn.com]()
- Intégrer des fonctionnalités à votre application via l'api: [api.pylearn.com]()
- Collaborer dans le but d'améliorer le code source du projet

## Si vous souhaitez intéragir avec l'IA via l'API

Nous vous prierons de bien vouloir consulter la documentation du projet via le [docs]()  
Le **endPoint** principal de la plateforme est servi sur l' **url**:
```
https://pylearn/api
```
Vous pouvez effectuer des requettes en **Post** avec comme label les informations suivantes:
```json
{
  "key": "your api key",
  "message": "message"
}
```
Si votre requette s'effectue bien vous recevrez une **reponse** suivant ce scéma:
```json
{
  "statut": true,
  "response": "Réponse de l'api",
  "suggestions": [],
  "WebSource": [] 
}
```

