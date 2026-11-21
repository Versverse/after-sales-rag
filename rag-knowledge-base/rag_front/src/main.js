import './assets/main.css'

import { createApp } from 'vue'
import App from './App.vue'
import { ElAvatar, ElButton, ElIcon, ElInput, ElUpload } from 'element-plus'
import 'element-plus/dist/index.css'

const app = createApp(App)

for (const component of [ElAvatar, ElButton, ElIcon, ElInput, ElUpload]) {
  app.use(component)
}

app.mount('#app')
