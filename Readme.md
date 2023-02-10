# Readme

The present text is a proposal to solve the problem presented in:

paper https://jcheminf.biomedcentral.com/articles/10.1186/s13321-021- 00541-z

In order to model the classification of molecules stored in the SMILE system, pyTorch was used together with the Deep Graph Library for graph neural networks.First, the available data was analyzed. Pre-opsed molecules were used.

| Activity | Smiles |
| --- | --- |
| 0 | CCOC(=O)C1(CCN(C)CC1)c1ccccc1
 |

The value "1" was considered a toxic parameter, and the value "0" was non-toxic.

The defined molecule was transformed into a graph using the RDKit library. Based on the characteristics of the molecule, a gaffe neural network was designed, which then consisted of two convolutional layers.For the created and renovated model, a simple REST API was built using the Flask library.

In the application folder by typing in the terminal:

$ flask --app main run

We can run the application.

Using the software for testing REST connections (Insomnia, Postman), you can verify the operation of the application by sending a JSON file using the POST method.

{

"smile":"C[N+](C)CCCN1c2ccccc2CCc2ccccc21"

}

And will recive:

{

"answear": "Toxic"

}

![](RackMultipart20230210-1-lputs6_html_e2f096a3c93c154c.png) ![](RackMultipart20230210-1-lputs6_html_7771b0c099ff8120.jpg)