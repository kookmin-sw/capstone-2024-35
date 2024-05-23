const { exec } = require('child_process');
const path = require('path');

exports.getsms = async (req, res) => {
    console.log('get sms test')
    // console.log(req.user)
    // console.log(req.body)
    res.send('get sms response')
};

exports.postsms = async (req, res) => {
    console.log('post sms test')
    console.log(req.user)
    console.log(req.body)
    res.send('post sms response')
};

exports.parseMessage = async (req, res) => {
    console.log('parseMessage')
    // res.send('python')
    // console.log(req.body)
    
    // res.send('dd')
    const params = JSON.stringify(req.body);
    // console.log(req.body.message)
    // console.log('post params: ', params)
    const scriptPath = path.join(__dirname, '../../DataExtract/src/script.py')
    exec(`python3 ${scriptPath} '${req.body.message}'`, (error, stdout, stderr) => {
        if (error) {
            console.error(`exec error: ${error}`);
            return res.status(500).send(error);
        }
        if (stderr) {
            console.error(`stderr: ${stderr}`);
            return res.status(500).send(stderr);
        }
        // res.send(stdout.trim());
        // console.log(stdout.trim())
        const result = stdout.trim()
        console.log(result)
        console.log('exec out')
    });

};
