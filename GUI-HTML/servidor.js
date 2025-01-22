const { MongoClient, ObjectId } = require('mongodb');
const cors = require('cors');
const express = require('express');
const { exec } = require('child_process');

const CryptoJS = require("crypto-js");
const fs = require('fs');
const path = require('path');
const moment = require('moment');
const session = require('express-session');
const cookieParser = require('cookie-parser');
const { spawn } = require('child_process');

const { PythonShell } = require('python-shell');

function calcularHashSHA256(data) {
    const hash = CryptoJS.SHA256(data).toString(CryptoJS.enc.Hex);
    return hash;
}

const app = express();

app.use(express.json());
app.use(cors());

const uri = 'mongodb+srv://fjmartinez216:fjmartinez216@cluster0.g9crmgr.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0';
const client = new MongoClient(uri, { useNewUrlParser: true, useUnifiedTopology: true });
var usuariosConectados = {};

const mongoose = require('mongoose');
const Schema = mongoose.Schema;

// Define el esquema para la colección de imágenes
const imagenSchema = new Schema({
  nombre: String,
  datos: Buffer, // Almacena los datos binarios de la imagen
});

// Crea el modelo para la colección de imágenes
const Imagen = mongoose.model('Imagen', imagenSchema);




/*
async function insertarImagen(nombreImagen, rutaImagen, denominacion) {
    try {
  
      // Seleccionar la base de datos "tfg"
      const database = client.db('tfg');
  
      // Seleccionar la colección "imagenes"
      const imagenesCollection = database.collection('moleculas');
  
      // Leer los datos binarios de la imagen desde el archivo
      const datosBuffer = fs.readFileSync(rutaImagen);

      const fechaEspecifica = moment('2024-04-14');

        // Formatear la fecha al formato DD/MM/YYYY
      const fechaFormateada = fechaEspecifica.format('DD/MM/YYYY');
  
      // Crear un documento con el nombre de la imagen y los datos binarios
      const nuevaImagen = {
        _id: nombreImagen,
        datos: datosBuffer,
        denominacion: denominacion,
        tiempo: fechaFormateada,
      };
  
      // Insertar el documento en la colección "imagenes"
      await imagenesCollection.insertOne(nuevaImagen);
  
      console.log(`Imagen "${nombreImagen}" insertada correctamente en la colección "imagenes".`);
    } catch (error) {
      console.error('Error al insertar la imagen:', error);
    }
  }

  insertarImagen("C00507", "../GUI-HTML/imagenes/C00507.png", "Pentahidroxihexano");
  insertarImagen("C00508", "../GUI-HTML/imagenes/C00508.png", "Ribosa");
  insertarImagen("C01099", "../GUI-HTML/imagenes/C01099.png", "Ácido 6-fosfogluconato");
  insertarImagen("C04371", "../GUI-HTML/imagenes/C04371.png", "Dimetildisulfuro");
  insertarImagen("C08257", "../GUI-HTML/imagenes/C08257.png", "Glucosa");
  insertarImagen("C12988", "../GUI-HTML/imagenes/C12988.png", "Ácido Aspártico");
  insertarImagen("C18988", "../GUI-HTML/imagenes/C18988.png", "Adenosina monofosfato");
  insertarImagen("DB09165", "../GUI-HTML/imagenes/DB09165.png", "Tetrahidroxidodifosfina trióxido");
  insertarImagen("DIADB099", "../GUI-HTML/imagenes/DIADB099.png", "Ácido acetilsalicílico");
  */


const ListaStringSchema = new Schema({
    nombres: [String]
});

const ListaString = mongoose.model('ListaString', ListaStringSchema);



/*
async function guardarListaStrings() {
    try {
      const database = client.db('tfg');
      const asociacionesCollection = database.collection('asociaciones');
      var listaImagenes = ["C00001", "C00007", "C00009", "C00011", "C00013", "C00018", "C00022"];
  
      const documento = { _id: "root", nombres: listaImagenes };
  
      const resultado = await asociacionesCollection.insertOne(documento);
    } catch (error) {
      console.error('Error al guardar la lista de strings en MongoDB:', error);
    } finally {
    }
}*/

/*
    async function recuperarYGuardarImagen(nombreImagen, rutaGuardar) {
        const client = new MongoClient(uri, { useNewUrlParser: true, useUnifiedTopology: true });
    
        try {
        // Conectar a MongoDB Atlas
        await client.connect();
    
        const database = client.db('tfg');
        const imagenesCollection = database.collection('imagenes');
    
        // Busca la imagen por nombre en la base de datos
        const imagen = await imagenesCollection.findOne({ nombre: nombreImagen });
    
        if (!imagen) {
            console.log(`No se encontró la imagen "${nombreImagen}".`);
            return;
        }
    
        // Obtiene los datos binarios como un Buffer desde el objeto Binary
        const datosBinarios = imagen.datos.buffer instanceof Buffer
            ? imagen.datos.buffer
            : Buffer.from(imagen.datos.buffer);
    
        // Guarda los datos binarios como un archivo de imagen en el sistema de archivos local
        fs.writeFileSync(rutaGuardar, datosBinarios);
        console.log(`Imagen "${nombreImagen}" guardada correctamente en "${rutaGuardar}".`);
        } catch (error) {
        console.error('Error al recuperar y guardar la imagen:', error);
        } finally {
        // Cierra la conexión a MongoDB Atlas
        await client.close();
        }
    }
    */


async function connectToMongoDB() {
    try {
        await client.connect();
        console.log('Conectado a MongoDB');
    } catch (error) {
        console.error('Error conectando a MongoDB:', error);
        process.exit(1);
    }
}

connectToMongoDB();

function generarIdSesionAleatorio() {
  var resultado = "";
  var caracteres = "0123456789";
  var longitud = 8;

  for (var i = 0; i < longitud; i++) {
      var indiceAleatorio = Math.floor(Math.random() * caracteres.length);
      resultado += caracteres.charAt(indiceAleatorio);
  }

  return resultado;
}
app.use(cookieParser());
app.use(session({
    secret: 'secreto', // Se utiliza para firmar la cookie de sesión
    resave: false,
    saveUninitialized: false,
    cookie: {
        secure: false,
        maxAge: 24 * 60 * 60 * 1000
    }
}));

function devolverUsuarioAsociadoASesion(jugador){
  var nombre = usuariosConectados[jugador];
  return nombre; 
}


function requireAuth(req, res, next) {
    const idSesion = req.headers.cookie;
    const numeroCookie = idSesion.split('=')[1];


    if (numeroCookie) {
        if(usuariosConectados[numeroCookie.toString()] !== undefined){
            next();
        }
        else{
            res.redirect('/login.html');
        }
    } else {
        res.redirect('/login.html');
    }
}

app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'login.html'));
});

app.get('/gestion.html', requireAuth,  (req, res) => {
    res.sendFile(path.join(__dirname, 'gestion.html'));
});
app.get('/prediccion.html', requireAuth,  (req, res) => {
    res.sendFile(path.join(__dirname, 'prediccion.html'));
});
app.get('/resultado.html', requireAuth,  (req, res) => {
    res.sendFile(path.join(__dirname, 'resultado.html'));
});
app.get('/login.html', (req, res) => {
    res.sendFile(path.join(__dirname, 'login.html'));
});

app.use(express.static(__dirname));

app.post('/api/comprobarLogin', async (req, res) => {
  const { usuario, pass} = req.body;
  const hash = calcularHashSHA256(pass);
  try {
      const database = client.db('tfg');
      const usuarios = database.collection('usuarios');

      const usuarioExistente = await usuarios.findOne({usuario: usuario, pass: hash});

      if (usuarioExistente) {
          var idSesion = generarIdSesionAleatorio();
          while(usuariosConectados[idSesion] !== undefined){
            var idSesion = generarIdSesionAleatorio();
          }
          usuariosConectados[idSesion] = usuario;

          res.send({ idSesion: idSesion});
      } else {
          res.send({ idSesion: null});
      }
  } catch (error) {
      console.error('Error al comprobar el login:', error);
      res.status(500).send('Error al manejar el login.');
  }
});

app.post('/api/recuperarGestion', async (req, res) => {
    const {usuario} = req.body;
    try {
        const database = client.db('tfg');
        const asociacionesCollection = database.collection('asociaciones');
        const moleculasCollection = database.collection('moleculas');

        const conjuntoAsociaciones = await asociacionesCollection.findOne({_id: usuario});

        const busqueda = { _id: { $in: conjuntoAsociaciones.nombres } };

        const proyeccion = {datos: 0};
        const resultados = await moleculasCollection
            .find(busqueda)
            .project(proyeccion)
            .toArray();

        res.send({moleculas: resultados});

    } catch (error) {
        console.error('Error al comprobar el login:', error);
        res.status(500).send('Error al manejar el login.');
    }
});

app.post('/api/recuperarPrediccion', async (req, res) => {
    const {usuario} = req.body;
    try {
        const database = client.db('tfg');
        const asociacionesCollection = database.collection('asociaciones');
        const moleculasCollection = database.collection('moleculas');

        const conjuntoAsociaciones = await asociacionesCollection.findOne({_id: usuario});

        const busqueda = { _id: { $in: conjuntoAsociaciones.nombres } };

        const proyeccion = {_id: 1, datos: 1};
        const resultados = await moleculasCollection
            .find(busqueda)
            .project(proyeccion)
            .toArray();


        res.send({moleculas: resultados});

    } catch (error) {
        console.error('Error al comprobar el login:', error);
        res.status(500).send('Error al manejar el login.');
    }
});


app.post('/api/recuperarTodos', async (req, res) => {
    const {usuario} = req.body;
    try {
        const database = client.db('tfg');
        const moleculasCollection = database.collection('moleculas');

        const busqueda = {};

        const proyeccion = {datos: 0};
        const resultados = await moleculasCollection
            .find(busqueda)
            .project(proyeccion)
            .toArray();


        res.send({moleculasTotales: resultados});

    } catch (error) {
        console.error('Error al comprobar el login:', error);
        res.status(500).send('Error al manejar el login.');
    }
});

app.post('/api/agregarMoleculaGestion', async (req, res) => {
    const {molecula, usuario} = req.body;
    try {
        const database = client.db('tfg');
        const asociacionesCollection = database.collection('asociaciones');


        await asociacionesCollection.updateOne(
            { _id: usuario },
            { $push: { nombres: molecula} }
        );

        res.send({});

    } catch (error) {
        console.error('Error al comprobar el login:', error);
        res.status(500).send('Error al manejar el login.');
    }
});

app.post('/api/eliminarMoleculaGestion', async (req, res) => {
    const {molecula, usuario} = req.body;
    try {
        const database = client.db('tfg');
        const asociacionesCollection = database.collection('asociaciones');


        await asociacionesCollection.updateOne(
            { _id: usuario },
            { $pull: { nombres: molecula} }
        );

        res.send({});

    } catch (error) {
        console.error('Error al comprobar el login:', error);
        res.status(500).send('Error al manejar el login.');
    }
});

app.post('/api/realizarPrediccion', async (req, res) => {
    const {imagenes, moleculaSeleccionada} = req.body;
    try {
        let imagenesJSON = JSON.stringify(imagenes);
        //'C:\\Users\\Home\\AppData\\Local\\Programs\\Python\\Python37\\python.exe'
        const pythonProcess = spawn('python', ['../GUI-HTML/py/predicciones.py', moleculaSeleccionada]);

        let resultados = ''; 

        pythonProcess.stdout.on('data', (data) => {
            resultados += data.toString(); // Concatenar la salida del proceso
        });

        pythonProcess.stderr.on('data', (data) => {
            console.error(`Error en el script Python: ${data.toString()}`);
        });

        pythonProcess.stdin.write(imagenesJSON);
        pythonProcess.stdin.end();

        pythonProcess.on('exit', (code, signal) => {
            if (code !== 0) {
                console.error(`Proceso hijo finalizó con código de salida ${code} y señal ${signal}`);
                res.status(500).send(`Error al enviar datos. Código de salida: ${code}. Señal: ${signal}`);
            } else {
                try {
                    const parsedResultados = JSON.parse(resultados);
    
                    res.status(200).json(parsedResultados);
                } catch (error) {
                    console.error('Error al parsear resultados:', error);
                    res.status(500).send('Error al procesar los resultados');
                }
            }
        });

  } catch (error) {
    console.error('Error al enviar datos:', error.message);
    res.status(500).send('Error al enviar datos');
  }
});




app.listen(3000, () => {
    console.log('Servidor escuchando en el puerto 3000');
});

